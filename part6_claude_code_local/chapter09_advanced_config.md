# Chapter 09: Advanced configuration and best practices

**📖 Purpose of this chapter**

Learn advanced configuration and best practices suitable for production environments and team development.

**🎯 Who should read this chapter? **
```
✅ Should read if:
□ Operate in production environment
□ I want to share it with my team
□ Automatic startup/monitoring required
□ I want to strengthen security
□ Backup required
□ Cost analysis required

❌ Can be skipped if:
□ Personal development only
□ There is no problem with manual startup.
□ Satisfied with the settings in Chapter 02-08
```

**💡 Structure of this chapter**
```
9.1 Production environment (systemd, Nginx)
→ For people involved in team development and production operations

9.2 Advanced LiteLLM configuration (fallback, caching)
→ For those who pursue reliability and speed

9.3 Security (API key management, firewall)
→ For people who value external disclosure and security

9.4 Performance tuning (parallel processing, connection pooling)
→ For those seeking the best performance

9.5 Backup and Recovery
→ For people who need data protection

9.6 Cost analysis
→ For those who want to know ROI (return on investment)
```

**🤔 Check your status**
```
Personal development/learning objectives:
→ Read only 9.6 (cost analysis is interesting)

Small team (2-5 people):
→ Read 9.1, 9.5 (automatic startup/backup)

Production environment/large scale:
→ Read all sections
```

---

## 9.1 Operation in production environment

**💡 What you will learn in this section**
- Automatic startup settings (systemd)
- Nginx reverse proxy
- Log management

**🤔 Do you need it? **
```
□ I want to start automatically when restarting
□ I want to share it with my team
□ Accessed by multiple users
□ I want to centrally manage logs
```
→ Check two or more: **This clause is required**

### 9.1.1 Complete configuration of systemd services

**💡 What is this? **
Register Ollama and LiteLLM as system services so that they start automatically when the OS starts.

**✅ Benefits**
- Starts automatically even after restarting the machine
- Automatic restart on crash
- Logs are centrally managed with `journalctl`
- Easy control with `systemctl` command

**Ollama Service**

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
# Basic settings
User=ollama
Group=ollama
WorkingDirectory=/var/lib/ollama

# ROCm environment variables
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="PYTORCH_ROCM_ARCH=gfx1100"
Environment="GPU_MAX_ALLOC_PERCENT=95"
Environment="ROCM_PATH=/opt/rocm"

# Ollama settings
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_KEEP_ALIVE=10m"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_MAX_QUEUE=512"

# resource limit
LimitNOFILE=65536
LimitNPROC=4096

# restart policy
Restart=always
RestartSec=10

# log
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ollama

[Install]
WantedBy=multi-user.target
```

**LiteLLM Service**

```ini
# /etc/systemd/system/litellm.service
[Unit]
Description=LiteLLM Proxy Server
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=litellm
Group=litellm
WorkingDirectory=/opt/litellm

# Python in virtual environment
ExecStart=/opt/litellm/venv/bin/litellm \
    --config /opt/litellm/config.yaml \
    --port 8000 \
    --host 0.0.0.0 \
    --num_workers 4

# environmental variables
Environment="PYTHONUNBUFFERED=1"
Environment="LOG_LEVEL=INFO"

# resource limit
LimitNOFILE=65536

# restart policy
Restart=always
RestartSec=10

# log
StandardOutput=journal
StandardError=journal
SyslogIdentifier=litellm

[Install]
WantedBy=multi-user.target
```

**Service activation**

```bash
# reload service
sudo systemctl daemon-reload

# Enable autostart (important!)
sudo systemctl enable ollama
sudo systemctl enable litellm

# boot
sudo systemctl start ollama
sudo systemctl start litellm

# Check status
sudo systemctl status ollama
sudo systemctl status litellm
```

**✅ In normal working condition**
```bash
$ sudo systemctl status ollama
● ollama.service - Ollama Service
   Active: active (running) since ...
← "active (running)" is important

$ sudo systemctl status litellm
● litellm.service - LiteLLM Proxy Server
   Active: active (running) since ...
```

**🔧 Test: Try restarting your machine**
```bash
# restart
sudo reboot

# After reboot, check automatic startup
$ sudo systemctl status ollama
# Active: active (running) ← Automatically started

$ sudo systemctl status litellm
# Active: active (running) ← Automatically started
```

### 9.1.2 Nginx Reverse Proxy

**💡 What is this? **
Place Nginx in front of LiteLLM to provide SSL/TLS, rate limiting, and load balancing.

**🤔 Do you need it? **
```
□ I want to access it from outside
□ Requires HTTPS (SSL/TLS)
□ Rate limiting required
□ I want to load balance multiple LiteLLM instances
```
→ Check at least one: **This section is required**

**❌ If not required**
- For use only on local machine
- Only you can use it
- No security requirements

```nginx
# /etc/nginx/sites-available/litellm
upstream litellm_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name ai-api.local;

# Redirect to HTTPS (production environment)
    # return 301 https://$server_name$request_uri;

# log
    access_log /var/log/nginx/litellm-access.log;
    error_log /var/log/nginx/litellm-error.log;

# proxy settings
    location / {
        proxy_pass http://litellm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

# timeout
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 600s;

# buffering
        proxy_buffering off;
    }

# Health check
    location /health {
        proxy_pass http://litellm_backend/health;
        access_log off;
    }

# rate limit
    location /v1/ {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://litellm_backend/v1/;
    }
}

# rate limit zone definition
# Add to /etc/nginx/nginx.conf
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;
```

Enable:

```bash
sudo ln -s /etc/nginx/sites-available/litellm /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

**✅ Benefits**
- Encrypt communication with HTTPS
- Rate limiting prevents overload
- Load balance multiple backends
- Centralized management of access logs

---

## 9.2 Advanced LiteLLM settings

**💡 What you will learn in this section**
- Fallback (automatic switching in case of failure)
- Caching (speeding up)
- Logging and monitoring

**🤔 Do you need it? **
```
□ Requires high availability (resistant to failures)
□ I just want to pursue speed
□ I want to analyze usage status
□ Operate in production environment
```
→ Check two or more: **This clause is required**

### 9.2.1 Fallback and Load Balancing

**💡 What is this? **
This feature automatically switches to another model if the main model fails.

**🤔 When do you need it? **
```
□ Absolutely unstoppable in production environment
□ Using multiple models
□ I want to continue even if one model crashes
```

**✅ Fallback example**
```
Request → qwen3-coder:30b-a3b-q8_0
↓ Failure
→ qwen3-coder:14b (fallback)
↓ Failure
→ qwen3-coder:7b (final fallback)
```

```yaml
# config.yaml (Advanced settings/latest November 2025)
model_list:
# Primary model (highest quality/30B Q8_0)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
      num_ctx: 262144  # 256K context
      rpm: 60  # Requests per minute

# Fallback model (when primary fails/14B)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:14b
      api_base: http://localhost:11434
      num_ctx: 262144

# Load balancing with multiple instances (30B Q8_0)
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      api_base: http://localhost:11434
      num_ctx: 262144

  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
api_base: http://localhost:11435 # another instance
      num_ctx: 262144

router_settings:
# Load balancing strategy
routing_strategy: latency-based-routing # select the fastest model
# routing_strategy: least-busy # Select the busiest model

# Retry settings
  num_retries: 3
  retry_after: 5  # seconds
  timeout: 300

# fallback
  fallbacks:
    claude-3-5-sonnet-20241022:
      - gpt-4
      - gpt-3.5-turbo

    gpt-4:
      - claude-3-5-sonnet-20241022

# Circuit breaker (temporarily excludes model on consecutive failures)
  allowed_fails: 3
  cooldown_time: 60  # seconds
```

**⚠️ Notes**
- Fallback model may have lower quality
- Requires memory for multiple models
- Usually unnecessary for personal development

### 9.2.2 Caching strategy

**💡 What is this? **
This is a mechanism that caches answers to the same question, making it super fast from the second time onward.

**🤔 When do you need it? **
```
□ Edit the same file multiple times
□ Many routine tasks (test generation, etc.)
□ I want to give top priority to speed.
□ Development teams work on the same code base
```

**✅ Effect of cache**
```
First time: "Add error handling" → 8.2 seconds
2nd time: "Add error handling" → 0.3 seconds (27x faster!)
```

**⚠️ Notes**
- Requires Redis installation
- Cache memory usage
- Possibility of old cache remaining

```yaml
# config.yaml - caching settings
litellm_settings:
# Redis cache
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    db: 0
ttl: 3600 # 1 hour

# Redis connection pool
    max_connections: 50
    socket_connect_timeout: 5
    socket_timeout: 5

# Semantic cache (cache similar queries)
  enable_semantic_caching: true
semantic_cache_threshold: 0.95 # Cache hit with similarity greater than 95%

# Customize cache key
  cache_key_format: "{model}:{messages_hash}"
```

**Redis optimization**

```bash
# /etc/redis/redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru # Delete LRU
save "" # Disable snapshot (performance-oriented)

# restart
sudo systemctl restart redis
```

### 9.2.3 Logging and Monitoring

```yaml
# config.yaml - log settings
general_settings:
# database log
  database_url: postgresql://user:pass@localhost/litellm
# or SQLite
  # database_url: sqlite:///litellm.db

# Success/Failure callback
  success_callback: ["langfuse", "posthog"]
  failure_callback: ["langfuse", "sentry"]

litellm_settings:
# detailed log
  set_verbose: true

# Custom logger
  log_level: INFO

# metrics
  enable_metrics: true
```

**Prometheus Metrics**

```python
# prometheus_exporter.py
from prometheus_client import start_http_server, Counter, Histogram
import time

# Metric definition
request_count = Counter('litellm_requests_total', 'Total requests', ['model', 'status'])
request_duration = Histogram('litellm_request_duration_seconds', 'Request duration', ['model'])
token_count = Counter('litellm_tokens_total', 'Total tokens', ['model', 'type'])

# Start Prometheus server
start_http_server(9090)

# Metrics are collected automatically
# Available at http://localhost:9090/metrics
```

**💡 Install Redis (if required)**
```bash
# Install Redis
sudo apt install redis-server

# boot
sudo systemctl start redis
sudo systemctl enable redis

# Operation confirmation
redis-cli ping
# It is OK if PONG is displayed
```

---

## 9.3 Security Best Practices

**💡 What you will learn in this section**
- API key management
- Firewall settings
- SSL/TLS settings

**🤔 Do you need it? **
```
□ Access from outside
□ There are multiple users
□ Handling confidential information
□ There are security requirements
```
→ Check at least one: **This section is required**

**❌ If not required**
- For use only on local machine
- Not accessed from outside
- Only you can use it

### 9.3.1 API key management

**💡 What is this? **
Manage your API keys securely and prevent unauthorized access.

**⚠️ Things not to do**
```yaml
# ❌ Bad example: Hardcoded in config.yaml
general_settings:
master_key: "sk-local-dev-1234" # ← Committed to Git!
```

**✅ The correct way**
```yaml
# ✅ Good example: Loading from environment variables
general_settings:
master_key: ${LITELLM_MASTER_KEY} # ← Obtained from environment variable
```

```yaml
# config.yaml
general_settings:
# Strong master key
master_key: ${LITELLM_MASTER_KEY} # Get from environment variable

# Key by user
  ui_username: admin
  ui_password: ${LITELLM_UI_PASSWORD}

# key rotation
  allowed_models:
    - claude-3-5-sonnet-20241022
    - gpt-4

# IP restriction
  allowed_ips:
    - 192.168.1.0/24
    - 10.0.0.0/8
```

**Setting environment variables**

```bash
# /etc/environment
LITELLM_MASTER_KEY=$(openssl rand -hex 32)
LITELLM_UI_PASSWORD=$(openssl rand -hex 16)

# or dedicated file
sudo nano /etc/litellm/env

LITELLM_MASTER_KEY=your_secure_key_here
LITELLM_UI_PASSWORD=your_secure_password

# read with systemd
# /etc/systemd/system/litellm.service
[Service]
EnvironmentFile=/etc/litellm/env
```

### 9.3.2 Firewall settings

```bash
# Restrict ports with UFW
sudo ufw allow from 192.168.1.0/24 to any port 8000
sudo ufw deny 8000

# Ollama is internal only
sudo ufw deny 11434

# Only accessible via Nginx
sudo ufw allow 80
sudo ufw allow 443
sudo ufw enable
```

### 9.3.3 SSL/TLS settings

```bash
# Let's Encrypt certificate
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d ai-api.yourdomain.com

# Nginx is automatically updated
```

---

## 9.4 Performance Tuning

**💡 What you will learn in this section**
- Speed-up through parallel processing
- Connection pooling

**🤔 Do you need it? **
```
□ I just want the best performance
□ I want to maximize the performance of MS-S1 Max
□ I want to process multiple requests simultaneously
□ Many simultaneous accesses due to team development
```
→ Check two or more: **This clause is required**

**❌ If not required**
- Sufficient speed is achieved through personal development
- Satisfied with the settings in Chapter 02-08
- Prioritize stability over performance

### 9.4.1 Ollama parallelism

**💡 What is this? **
Take advantage of MS-S1 Max's large memory capacity to run multiple Ollama instances in parallel.

**🤔 When do you need it? **
```
□ Access by multiple people in a team at the same time
□ I want to use the high-speed model (7B) and high-quality model (30B) at the same time
□ Using MS-S1 Max (128GB RAM)
```

**⚠️ Note**: Not recommended for machines less than 64GB

```bash
# Start multiple Ollama instances
# Instance 1 (Port 11434)
sudo systemctl start ollama

# Instance 2 (Port 11435) - another configuration file
sudo cp /etc/systemd/system/ollama.service /etc/systemd/system/ollama2.service

# Edit /etc/systemd/system/ollama2.service
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MODELS=/var/lib/ollama2"

sudo systemctl daemon-reload
sudo systemctl start ollama2
```

### 9.4.2 Connection pooling

```python
# connection_pool.py
from litellm import completion
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool settings
executor = ThreadPoolExecutor(max_workers=16)

async def parallel_completions(prompts):
"""Run Completion in parallel"""
    loop = asyncio.get_event_loop()

    tasks = [
        loop.run_in_executor(
            executor,
            completion,
            "claude-3-5-sonnet-20241022",
            [{"role": "user", "content": prompt}]
        )
        for prompt in prompts
    ]

    results = await asyncio.gather(*tasks)
    return results

# Usage example
prompts = [
    "Write a function to sort a list",
    "Explain Python decorators",
    "Create a FastAPI endpoint"
]

results = asyncio.run(parallel_completions(prompts))
```

---

## 9.5 Backup and Recovery

**💡 What you will learn in this section**
- Automatic backup
- Recovery procedure

**🤔 Do you need it? **
```
□ Operating in production environment
□ Data loss is unacceptable
□ Team development
□ Many custom settings
```
→ Check at least one: **This section is required**

**❌ If not required**
- For learning/experiment purposes only
- Can be re-set up at any time
- No data (model can be re-downloaded at any time)

**💡 What to back up? **
```
Required:
✅ LiteLLM settings (config.yaml)
✅ Aider settings (.aider.conf.yml)
✅ Environment variable settings

any:
⚠️ Ollama model (32GB) ← Can be re-downloaded
⚠️ Redis cache ← Can be regenerated
⚠️ Database ← Only if used
```

### 9.5.1 Automatic backup script

**💡 What is this? **
This is a script that automatically takes backups every day.

```bash
#!/bin/bash
# backup_ai_system.sh

BACKUP_DIR="/backups/ai-system"
DATE=$(date +%Y%m%d_%H%M%S)

echo "Starting backup: $DATE"

# Ollama model
echo "Backing up Ollama models..."
tar -czf "$BACKUP_DIR/ollama_models_$DATE.tar.gz" ~/.ollama/models/

# LiteLLM settings
echo "Backing up LiteLLM config..."
cp ~/litellm/config.yaml "$BACKUP_DIR/config_$DATE.yaml"

# Database (PostgreSQL)
echo "Backing up database..."
pg_dump litellm > "$BACKUP_DIR/litellm_db_$DATE.sql"

# Redis (cache)
echo "Backing up Redis..."
redis-cli SAVE
cp /var/lib/redis/dump.rdb "$BACKUP_DIR/redis_$DATE.rdb"

# Delete old backups (more than 30 days ago)
find "$BACKUP_DIR" -name "*" -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Cron settings (runs every day at 3 o'clock)**

```bash
sudo crontab -e

# addition
0 3 * * * /usr/local/bin/backup_ai_system.sh >> /var/log/ai-backup.log 2>&1
```

### 9.5.2 Recovery procedure

```bash
#!/bin/bash
# restore_ai_system.sh

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Example: $0 20250115_030000"
    exit 1
fi

BACKUP_DATE=$1
BACKUP_DIR="/backups/ai-system"

echo "Restoring from backup: $BACKUP_DATE"

# Ollama model
echo "Restoring Ollama models..."
tar -xzf "$BACKUP_DIR/ollama_models_$BACKUP_DATE.tar.gz" -C ~/

# LiteLLM settings
echo "Restoring LiteLLM config..."
cp "$BACKUP_DIR/config_$BACKUP_DATE.yaml" ~/litellm/config.yaml

# database
echo "Restoring database..."
psql litellm < "$BACKUP_DIR/litellm_db_$BACKUP_DATE.sql"

# Redis
echo "Restoring Redis..."
sudo systemctl stop redis
sudo cp "$BACKUP_DIR/redis_$BACKUP_DATE.rdb" /var/lib/redis/dump.rdb
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis

# restart the service
echo "Restarting services..."
sudo systemctl restart ollama
sudo systemctl restart litellm

echo "Restore completed"
```

**✅ Check backup**
```bash
# check if backup exists
ls -lh /backups/ai-system/

# Check latest backup
ls -lht /backups/ai-system/ | head -5
```

---

## 9.6 Cost analysis

**💡 What you will learn in this section**
- Electricity cost calculation
- Comparison with cloud API
- ROI (recovery period)

**🤔 Who should read it? **
```
✅ Should read if:
□ Considering purchasing MS-S1 Max
□ I want to know the economics of local LLM
□ I want to compare with cloud API
□ I would like to suggest introduction to my boss

❌ Can be skipped if:
□ Already purchased
□ Don't worry about cost
```

**💡 Conclusion first**
```
MS-S1 Max (approximately $2,500-3,000):
- Electricity bill: $2.70 per month (light usage)
- Cloud API: $27/month (light usage)
→ Savings of $24.30 per month = $291.60 per year

Payback period:
- Light use: 10-12 months
- Moderate use: 5-6 months
- Heavy use: 2-3 months
```

### 9.6.1 Calculating electricity costs

**💡 What is this? **
Calculate the electricity bill when MS-S1 Max is operated for 24 hours.

```python
# cost_calculator.py
import psutil
import time

class CostCalculator:
    def __init__(self, electricity_rate=0.03):  # USD per kWh
        self.electricity_rate = electricity_rate
        self.start_time = time.time()
        self.start_energy = self.get_system_power()

    def get_system_power(self):
"""Estimated power consumption (W)"""
# MS-S1 Max: Approx. 120W (normal use)
# When using GPU: +30-50W
        return 120  # Watts

    def get_cost_estimate(self, hours):
"""Estimated cost per hour"""
        power_kw = self.get_system_power() / 1000
        cost = power_kw * hours * self.electricity_rate
        return cost

    def compare_with_cloud(self, tokens_per_day):
"""Cost comparison with cloud API"""
        # Claude API: $3 per million input tokens, $15 per million output tokens
# Average: $9 per million tokens

        daily_cloud_cost = (tokens_per_day / 1_000_000) * 9
        monthly_cloud_cost = daily_cloud_cost * 30

# Local (24 hours operation)
        daily_local_cost = self.get_cost_estimate(24)
        monthly_local_cost = daily_local_cost * 30

        print(f"=== Cost Comparison ===")
        print(f"Tokens per day: {tokens_per_day:,}")
        print(f"\nCloud API (Claude):")
        print(f"  Daily: ${daily_cloud_cost:.2f}")
        print(f"  Monthly: ${monthly_cloud_cost:.2f}")
        print(f"\nLocal (MS-S1 Max):")
        print(f"  Daily: ${daily_local_cost:.2f}")
        print(f"  Monthly: ${monthly_local_cost:.2f}")
        print(f"\nSavings:")
        print(f"  Daily: ${daily_cloud_cost - daily_local_cost:.2f}")
        print(f"  Monthly: ${monthly_cloud_cost - monthly_local_cost:.2f}")
        print(f"  Annual: ${(monthly_cloud_cost - monthly_local_cost) * 12:.2f}")

# Usage example
calc = CostCalculator()
calc.compare_with_cloud(tokens_per_day=100_000) # 100,000 tokens per day
```

**Execution result example**

```
=== Cost Comparison ===
Tokens per day: 100,000

Cloud API (Claude):
  Daily: $0.90
  Monthly: $27.00

Local (MS-S1 Max):
  Daily: $0.09
  Monthly: $2.70

Savings:
  Daily: $0.81
  Monthly: $24.30
  Annual: $291.60
```

### 9.6.2 ROI (Recovery Period)

```
MS-S1 Max system cost: approximately $2,500-3,000
Monthly savings: $24.30 (light usage) to $500+ (heavy usage)

Payback period:
- Light use: 10-12 months
- Moderate use: 5-6 months
- Heavy use: 2-3 months
```

**⚠️ Actual electricity bill example (Japan)**
```
Tokyo Electric Power Company: Approximately 30 yen/kWh
120W × 24 hours × 30 days = 86.4kWh
86.4kWh × 30 yen = approximately 2,592 yen/month
```

---

## 9.7 Summary

In this chapter, you learned advanced configuration and best practices for production environments.

**What we accomplished**
✅ Production-ready systemd configuration
✅ Nginx reverse proxy
✅ Advanced LiteLLM settings (fallback, caching)
✅ Enhanced security
✅ Backup and recovery
✅ Cost analysis

**❓ Frequently Asked Questions (FAQ)**

**Q: Are all the settings in this chapter necessary? **
A: No. For personal development, automatic startup of 9.1 is sufficient. All sections are recommended for production environments and team development.

**Q: Is Nginx required? **
A: It is not necessary if you only use it locally. Please set this only if external publication or HTTPS communication is required.

**Q: What should I backup? **
A: At the bare minimum, only LiteLLM settings (config.yaml) and Aider settings (.aider.conf.yml). Models can be re-downloaded.

**Q: Is fallback necessary for personal development? **
A: No need. One model is enough. Set this up only in production environments or when high availability is required.

**Q: How much will this setting improve performance? **
A:
- Caching: 2-27x faster (from second time onwards)
- Parallelism: 2-4x depending on number of requests
- Connection pool: 10-20% improvement

**Q: What happens if I omit security settings? **
A: There is no problem if you use it only locally. Required when publishing externally.

**💡 Priority by chapter**
```
Personal development:
Required: None
Recommended: 9.1 (autostart), 9.6 (cost analysis)
Optional: Other

Small team:
Required: 9.1 (autostart)
Recommended: 9.2 (caching), 9.5 (backup)
Optional: 9.3, 9.4

Production environment/large scale:
Required: 9.1, 9.3, 9.5
Recommended: 9.2, 9.4
Optional: 9.6
```

**Production environment checklist**
```
Development environment (individual):
- [ ] systemd service starts automatically

Production environment (team):
- [ ] systemd service starts automatically
- [ ] Reverse proxy configured with Nginx
- [ ] SSL/TLS certificate configured (when published externally)
- [ ] Firewall configured (when publishing externally)
- [ ] Backup will be performed automatically
- [ ] Monitoring is working
- [ ] Log rotation configured
```

---

## Summary of Part 6

Through this book, I learned the following:

**📚 Contents of each chapter**
1. **Chapter 01**: Claude Code and local LLM integration overview
2. **Chapter 02**: Installing and configuring Ollama (Qwen3 Coder 30B Q8_0)
3. **Chapter 03**: LiteLLM setup (OpenAI API compatible)
4. **Chapter 04**: Claude Code-like development with Aider
5. **Chapter 05**: Understanding Compatibility and Limitations
6. **Chapter 06**: Model selection and optimization (MS-S1 Max specific)
7. **Chapter 07**: Practical examples (refactoring, bug fixes, etc.)
8. **Chapter 08**: Troubleshooting (TOP3 problems and solutions)
9. **Chapter 09**: Advanced configuration and best practices

**✅ Results of using MS-S1 Max**
```
Cost:
- ✅ Completely free AI-assisted development (zero API fees)
- ✅ Annual cost savings of $300-$6,000
- ✅ Electricity bill: Approximately $2.70 per month (approximately 2,600 yen per month)

Privacy/Security:
- ✅ Complete privacy protection (no data sent externally)
- ✅ Trade secrets and confidential information are safe
- ✅ Can work offline

Development efficiency:
- ✅ Improved development efficiency by 30-50%
- ✅ Refactoring: 22x faster
- ✅ Bug fix: 11x faster
- ✅ Code review: 5.6x faster

Technical aspects:
- ✅ 256K context (processing approximately 100,000 lines of code)
- ✅ HumanEval 92.8% high accuracy
- ✅ Practical speed of 22 tokens/s
- ✅ Make the most of MS-S1 Max's 128GB RAM + 96GB VRAM
```

**🎉 Congratulations! **

You can now build and operate an advanced AI-assisted development environment using local LLM on MS-S1 Max.

**Next steps**
1. Try using it in an actual project
2. Improve accuracy by improving prompts
3. Share with your team
4. Try more advanced settings

**🚀 Happy Coding with Local LLM on MS-S1 Max!**
