# Chapter 08: Monitoring and Troubleshooting

## 8.1 Log management

### 8.1.1 Structured logging

```python
# logging_config.py
import logging
import json
from datetime import datetime
from typing import Dict, Any
from pythonjsonlogger import jsonlogger

class CustomJsonFormatter(jsonlogger.JsonFormatter):
"""Custom JSON Formatter"""

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

# add custom field
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

# Environmental information
        log_record['environment'] = settings.app_env

def setup_logging(log_level: str = "INFO"):
"""Logging settings"""

# root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

# Console handler (JSON format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)

# File handler (rotation)
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    return root_logger

# Usage example
logger = setup_logging()

# log output
logger.info("Application started", extra={
    "user_id": "user123",
    "request_id": "req456"
})

logger.error("Database connection failed", extra={
    "error_code": "DB_CONN_001",
    "database": "postgres",
    "retry_count": 3
})
```

### 8.1.2 Request tracing

```python
# request_tracing.py
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class RequestTracingMiddleware(BaseHTTPMiddleware):
"""Request Tracing Middleware"""

    async def dispatch(self, request: Request, call_next: Callable):
# Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

# set log context
        start_time = time.time()

        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host
            }
        )

        try:
# process the request
            response = await call_next(request)

# Calculate processing time
            duration = time.time() - start_time

            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "duration": duration
                }
            )

# Add request ID to response header
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "duration": duration
                },
                exc_info=True
            )
            raise

app = FastAPI()
app.add_middleware(RequestTracingMiddleware)
```

### 8.1.3 Log aggregation (ELK stack)

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
      - ./logs:/logs
    ports:
      - "5044:5044"
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch-data:
```

**Logstash settings**

```ruby
# logstash/pipeline/logstash.conf
input {
  file {
    path => "/logs/app.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
# Parse timestamp
  date {
    match => [ "timestamp", "ISO8601" ]
    target => "@timestamp"
  }

# Add tag to error level log
  if [level] == "ERROR" {
    mutate {
      add_tag => [ "error" ]
    }
  }

# Add tag to logs with slow request processing time
  if [duration] {
    if [duration] > 5 {
      mutate {
        add_tag => [ "slow_request" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "app-logs-%{+YYYY.MM.dd}"
  }

# for debugging
  stdout {
    codec => rubydebug
  }
}
```

## 8.2 Performance Monitoring

### 8.2.1 APM（Application Performance Monitoring）

```python
# apm_integration.py
from elasticapm import Client
from elasticapm.contrib.starlette import ElasticAPM
from fastapi import FastAPI

# APM client settings
apm_config = {
    'SERVICE_NAME': 'local-ai-api',
    'SERVER_URL': 'http://localhost:8200',
    'ENVIRONMENT': settings.app_env,
    'CAPTURE_BODY': 'all',
    'TRANSACTION_SAMPLE_RATE': 1.0
}

apm_client = Client(apm_config)

app = FastAPI()
app.add_middleware(ElasticAPM, client=apm_client)

# custom transaction
from elasticapm import capture_span

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    with capture_span('ollama_inference', span_type='inference'):
# Ollama Reasoning
        response = ollama.chat(model=request.model, messages=request.messages)

    with capture_span('database_save', span_type='db'):
# save database
        save_to_db(response)

    return response
```

### 8.2.2 Custom metrics collection

```python
# custom_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
from typing import Callable

# Metric definition
inference_counter = Counter(
    'ai_inference_total',
    'Total AI inferences',
    ['model', 'status']
)

inference_duration = Histogram(
    'ai_inference_duration_seconds',
    'AI inference duration',
    ['model'],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60]
)

token_counter = Counter(
    'ai_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: prompt or completion
)

model_load_gauge = Gauge(
    'ai_model_loaded',
    'Whether model is loaded',
    ['model']
)

system_info = Info(
    'system_info',
    'System information'
)

# set system information
system_info.info({
    'cpu': 'AMD Ryzen AI Max+ 395',
    'gpu': 'Radeon 8060S',
    'memory': '128GB',
    'rocm_version': '6.4.2'
})

# Metrics collection decorator
def track_inference(model: str):
"""Track inference metrics"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"

            try:
                result = await func(*args, **kwargs)

# count the number of tokens
                if 'usage' in result:
                    token_counter.labels(
                        model=model,
                        type='prompt'
                    ).inc(result['usage'].get('prompt_tokens', 0))

                    token_counter.labels(
                        model=model,
                        type='completion'
                    ).inc(result['usage'].get('completion_tokens', 0))

                return result

            except Exception as e:
                status = "error"
                raise

            finally:
# record processing time
                duration = time.time() - start_time
                inference_duration.labels(model=model).observe(duration)
                inference_counter.labels(model=model, status=status).inc()

        return wrapper
    return decorator

# Usage example
@track_inference(model="qwen2.5:14b")
async def generate_text(prompt: str):
    response = ollama.chat(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response
```

## 8.3 System monitoring

### 8.3.1 Resource monitoring

```python
# resource_monitor.py
import psutil
import GPUtil
from typing import Dict
import subprocess
import re

class SystemMonitor:
"""System Resource Monitor"""

    @staticmethod
    def get_cpu_info() -> Dict:
"""Get CPU information"""
        return {
            "percent": psutil.cpu_percent(interval=1, percpu=False),
            "percent_per_core": psutil.cpu_percent(interval=1, percpu=True),
            "count": psutil.cpu_count(),
            "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }

    @staticmethod
    def get_memory_info() -> Dict:
"""Get memory information"""
        mem = psutil.virtual_memory()
        return {
            "total": mem.total,
            "available": mem.available,
            "used": mem.used,
            "percent": mem.percent,
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2)
        }

    @staticmethod
    def get_gpu_info() -> Dict:
"""Get GPU information (AMD ROCm)"""
        try:
# Get information using rocm-smi command
            result = subprocess.run(
                ['rocm-smi', '--showuse', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True
            )

# Parse (simple version)
            output = result.stdout

# GPU usage
            use_match = re.search(r'GPU use \(%\)\s+:\s+(\d+)', output)
            gpu_percent = int(use_match.group(1)) if use_match else 0

# VRAM memory
            vram_match = re.search(r'VRAM Total Memory \(B\)\s+:\s+(\d+)', output)
            vram_used_match = re.search(r'VRAM Total Used Memory \(B\)\s+:\s+(\d+)', output)

            vram_total = int(vram_match.group(1)) if vram_match else 0
            vram_used = int(vram_used_match.group(1)) if vram_used_match else 0

            return {
                "gpu_percent": gpu_percent,
                "vram_total": vram_total,
                "vram_used": vram_used,
                "vram_total_gb": round(vram_total / (1024**3), 2),
                "vram_used_gb": round(vram_used / (1024**3), 2),
                "vram_percent": round((vram_used / vram_total) * 100, 2) if vram_total > 0 else 0
            }

        except Exception as e:
            return {
                "error": str(e),
                "gpu_percent": 0,
                "vram_total": 0,
                "vram_used": 0
            }

    @staticmethod
    def get_disk_info() -> Dict:
"""Get disk information"""
        disk = psutil.disk_usage('/')
        return {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2)
        }

    @staticmethod
    def get_network_info() -> Dict:
"""Get network information"""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "mb_sent": round(net_io.bytes_sent / (1024**2), 2),
            "mb_recv": round(net_io.bytes_recv / (1024**2), 2)
        }

    @classmethod
    def get_all_stats(cls) -> Dict:
"""Get all system statistics"""
        return {
            "cpu": cls.get_cpu_info(),
            "memory": cls.get_memory_info(),
            "gpu": cls.get_gpu_info(),
            "disk": cls.get_disk_info(),
            "network": cls.get_network_info()
        }

# FastAPI endpoint
from fastapi import APIRouter

router = APIRouter()

@router.get("/system/stats")
async def get_system_stats():
"""Get system statistics"""
    return SystemMonitor.get_all_stats()
```

### 8.3.2 Prometheus Exporter

```python
# prometheus_exporter.py
from prometheus_client import Gauge
from threading import Thread
import time
from resource_monitor import SystemMonitor

# gauge metrics
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
memory_available = Gauge('system_memory_available_gb', 'Available memory in GB')

gpu_usage = Gauge('system_gpu_usage_percent', 'GPU usage percentage')
vram_usage = Gauge('system_vram_usage_percent', 'VRAM usage percentage')
vram_used = Gauge('system_vram_used_gb', 'VRAM used in GB')

disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')

class MetricsCollector:
"""System metrics collection"""

    def __init__(self, interval: int = 10):
        self.interval = interval
        self.running = False

    def start(self):
"""Start collection"""
        self.running = True
        thread = Thread(target=self._collect_loop, daemon=True)
        thread.start()

    def stop(self):
"""Stop collecting"""
        self.running = False

    def _collect_loop(self):
"""Metrics Collection Loop"""
        while self.running:
            try:
                stats = SystemMonitor.get_all_stats()

                # CPU
                cpu_usage.set(stats['cpu']['percent'])

# memory
                memory_usage.set(stats['memory']['percent'])
                memory_available.set(stats['memory']['available_gb'])

                # GPU
                if 'error' not in stats['gpu']:
                    gpu_usage.set(stats['gpu']['gpu_percent'])
                    vram_usage.set(stats['gpu']['vram_percent'])
                    vram_used.set(stats['gpu']['vram_used_gb'])

# disk
                disk_usage.set(stats['disk']['percent'])

            except Exception as e:
                print(f"Error collecting metrics: {e}")

            time.sleep(self.interval)

# start at startup
collector = MetricsCollector(interval=10)

@app.on_event("startup")
async def startup_event():
    collector.start()

@app.on_event("shutdown")
async def shutdown_event():
    collector.stop()
```

## 8.4 Alert settings

### 8.4.1 Prometheus Alert Rules

```yaml
# prometheus/alert_rules.yml
groups:
  - name: system_alerts
    interval: 30s
    rules:
# CPU usage alert
      - alert: HighCPUUsage
        expr: system_cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is {{ $value }}% (threshold: 80%)"

# Memory usage alert
      - alert: HighMemoryUsage
        expr: system_memory_usage_percent > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}% (threshold: 85%)"

# GPU usage alert
      - alert: HighGPUUsage
        expr: system_gpu_usage_percent > 90
        for: 10m
        labels:
          severity: info
        annotations:
          summary: "High GPU usage"
          description: "GPU usage is {{ $value }}% for 10 minutes"

# disk space alert
      - alert: LowDiskSpace
        expr: system_disk_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space"
          description: "Disk usage is {{ $value }}% (threshold: 80%)"

  - name: application_alerts
    interval: 30s
    rules:
# error rate alert
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} (threshold: 0.05)"

# response time alert
      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Slow response time"
          description: "95th percentile response time is {{ $value }}s"

# AI inference failure alert
      - alert: HighAIInferenceFailureRate
        expr: rate(ai_inference_total{status="error"}[5m]) / rate(ai_inference_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High AI inference failure rate"
          description: "Failure rate is {{ $value }} (threshold: 0.1)"
```

### 8.4.2 Alertmanager settings

```yaml
# alertmanager/config.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      continue: true
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'

  - name: 'critical-alerts'
    email_configs:
      - to: 'admin@example.com'
        from: 'alertmanager@example.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'alertmanager@example.com'
        auth_password: 'password'
        headers:
          Subject: '[CRITICAL] {{ .GroupLabels.alertname }}'

  - name: 'warning-alerts'
    webhook_configs:
      - url: 'http://localhost:5001/webhook'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

## 8.5 Troubleshooting

### 8.5.1 Common problems and solutions

**1. Ollama model is slow to load**

```bash
# diagnosis
ollama ps # Check running model
rocm-smi # Check GPU status

# Solution 1: Preload the model
ollama run qwen2.5:14b "test" &

# Solution 2: Optimize ROCm environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export GPU_MAX_ALLOC_PERCENT=95
```

**2. Out of memory error**

```python
# monitor_memory.py
import psutil
import gc

def check_memory():
"""Check memory usage"""
    mem = psutil.virtual_memory()
    print(f"Total: {mem.total / (1024**3):.2f} GB")
    print(f"Available: {mem.available / (1024**3):.2f} GB")
    print(f"Used: {mem.used / (1024**3):.2f} GB")
    print(f"Percent: {mem.percent}%")

    if mem.percent > 90:
        print("WARNING: Memory usage is high!")
# Execute garbage collection
        gc.collect()

# Check regularly
import threading

def memory_monitor():
    while True:
        check_memory()
        time.sleep(60)

threading.Thread(target=memory_monitor, daemon=True).start()
```

**3. Database connection error**

```python
# db_health_check.py
from sqlalchemy import create_engine, text
import time

def check_db_connection(database_url: str, max_retries: int = 5):
"""Check database connection"""
    for i in range(max_retries):
        try:
            engine = create_engine(database_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print(f"Database connection successful: {result.fetchone()}")
                return True
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries} failed: {e}")
time.sleep(2 ** i) # exponential backoff

    return False

# Usage example
if not check_db_connection(settings.database_url):
    print("ERROR: Failed to connect to database after retries")
```

**4. ROCm compatibility issue**

```bash
# ROCm diagnostic script
#!/bin/bash

echo "=== ROCm Diagnostic ==="

# ROCm version
echo "ROCm Version:"
rocm-smi --showversion

# GPU detection
echo -e "\nGPU Detection:"
rocminfo | grep "Name:" | head -1

# PyTorch ROCm support
python3 << EOF
import torch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"ROCm Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
EOF

# environmental variables
echo -e "\nEnvironment Variables:"
echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION}"
echo "PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}"
```

### 8.5.2 Debugging Tools

```python
# debug_tools.py
import traceback
import sys
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

def debug_function(func: Callable):
"""Decorator that outputs debugging information for functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling {func.__name__}")
        logger.debug(f"Args: {args}")
        logger.debug(f"Kwargs: {kwargs}")

        try:
            result = func(*args, **kwargs)
            logger.debug(f"Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise

    return wrapper

class PerformanceProfiler:
"""Performance Profiler"""

    def __init__(self):
        self.timings = {}

    def profile(self, name: str):
"""Profiling Decorator"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start

                if name not in self.timings:
                    self.timings[name] = []
                self.timings[name].append(duration)

                return result
            return wrapper
        return decorator

    def report(self):
"""Output profiling results"""
        print("\n=== Performance Profile ===")
        for name, timings in sorted(self.timings.items()):
            avg = sum(timings) / len(timings)
            min_time = min(timings)
            max_time = max(timings)
            print(f"{name}:")
            print(f"  Calls: {len(timings)}")
            print(f"  Avg: {avg:.4f}s")
            print(f"  Min: {min_time:.4f}s")
            print(f"  Max: {max_time:.4f}s")

# Usage example
profiler = PerformanceProfiler()

@profiler.profile("inference")
@debug_function
def run_inference(prompt: str):
    return ollama.chat(model="qwen2.5:14b", messages=[{"role": "user", "content": prompt}])

# report after execution
profiler.report()
```

## 8.6 Grafana Dashboard

### 8.6.1 Dashboard settings

```json
{
  "dashboard": {
    "title": "Local AI Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "AI Inference Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ai_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "System Resources",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU"
          },
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory"
          },
          {
            "expr": "system_gpu_usage_percent",
            "legendFormat": "GPU"
          }
        ],
        "type": "graph"
      },
      {
        "title": "VRAM Usage",
        "targets": [
          {
            "expr": "system_vram_used_gb",
            "legendFormat": "VRAM Used (GB)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

## 8.7 Summary

In this chapter, you learned about monitoring and troubleshooting in a production environment.

**Key points**

1. **Log management**
- Structured log (JSON format)
- Request tracing
- Aggregation by ELK stack

2. **Performance Monitoring**
- Detailed tracing with APM
- Prometheus metrics
- Custom metrics collection

3. **System Monitoring**
- CPU, memory, GPU monitoring
- ROCm specific metrics
- Resource usage tracking

4. **Alert**
   - Prometheus Alert Rules
- Notifications by Alertmanager
- Routing by importance

5. **Troubleshooting**
- Common problems and solutions
- Debugging tools
- Performance profiling

6. **Visualization**
- Grafana dashboard
- Real-time metrics
- Historical data analysis

In the next chapter, you will learn about operational best practices in a production environment.
