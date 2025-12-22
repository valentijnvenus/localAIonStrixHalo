# Chapter 4: Basic commands and usage

## 4.1 Basic structure of Ollama CLI

### 4.1.1 Command list

```bash
# Show help
ollama --help

# Main commands
ollama serve # start the server (usually autostart)
ollama run # run the model
ollama pull # download model
ollama push # Upload model
ollama list # local model list
ollama ps # Display running model
ollama cp # copy model
ollama rm # remove model
ollama show # Display model information
ollama create # create custom model
```

### 4.1.2 Basic grammar

```bash
# Basic format
ollama [command] [model] [options]

# example
ollama run llama3.1 "Hello"
ollama pull qwen2.5:7b
ollama list
```

## 4.2 Run the model

### 4.2.1 Interactive mode

```bash
# start interactive session
ollama run qwen2.5:7b

# prompt will be displayed
>>> Hello!
Hello! Is there anything I can do to help?

>>> Tell me about MS-S1 Max
MS-S1 Max is powered by AMD Ryzen AI Max+ 395 processor...

>>> /bye
```

**Special commands in interactive mode:**

| Command | Function |
|---------|------|
| `/bye` | End session |
| `/clear` | Clear conversation history |
| `/save [filename]` | Save conversation |
| `/load [filename]` | Load conversation |
| `/set parameter value` | Parameter change |
| `/show` | Display current settings |

### 4.2.2 One-shot execution

```bash
# specify direct prompt
ollama run llama3.1 "What is the capital of Japan?"

# output
The capital of Japan is Tokyo.

# multiline prompt
ollama run qwen2.5:14b "
Translate to Japanese:
Hello, how are you?
"
```

### 4.2.3 Executing from standard input

```bash
# input with pipe
echo "Explain quantum computing" | ollama run llama3.1

# input from file
cat prompt.txt | ollama run qwen2.5:7b

# heredoc
ollama run llama3.1 << EOF
Please summarize the following:
$(cat article.txt)
EOF
```

### 4.2.4 Parameter specification

```bash
# --verbose: Display detailed information
ollama run --verbose llama3.1 "Hello"

# --nowordwrap: Disable automatic line wrapping
ollama run --nowordwrap qwen2.5:7b "Long text..."

# --format json: Output in JSON format
ollama run --format json llama3.1 "List 3 colors"
```

## 4.3 Managing models

### 4.3.1 Downloading the model (pull)

```bash
# Download the latest version
ollama pull llama3.1

# specify a specific tag
ollama pull llama3.1:8b-instruct-q4_K_M

# Download multiple models
for model in llama3.1 qwen2.5:7b mistral; do
    ollama pull $model
done
```

**Show progress:**
```
pulling manifest
pulling 8cf58c9acf79... 100% ▕████████████████████████▏ 4.7 GB
pulling 8ab4849b038c... 100% ▕████████████████████████▏  249 B
pulling 23e0f4461c0c... 100% ▕████████████████████████▏  11 KB
verifying sha256 digest
writing manifest
success
```

### 4.3.2 Model list display (list)

```bash
# Local model list
ollama list

# Output example
NAME                              ID              SIZE    MODIFIED
llama3.1:latest                   abcd1234        4.7GB   2 hours ago
qwen2.5:7b                        efgh5678        4.9GB   1 day ago
qwen2.5:14b                       ijkl9012        9.2GB   2 days ago
mistral:latest                    mnop3456        4.1GB   3 days ago
```

### 4.3.3 Displaying model information (show)

```bash
# Model details information
ollama show llama3.1

# Output example
Model
  arch              llama
  parameters        8.0B
  quantization      Q4_K_M
  context length    131072
  embedding length  4096

Parameters
  stop    "<|start_header_id|>"
  stop    "<|end_header_id|>"
  stop    "<|eot_id|>"

License
  Meta Llama 3.1 Community License

System
  You are a helpful assistant.
```

```bash
# Display Modelfile
ollama show --modelfile llama3.1

# output
FROM /path/to/model
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
```

### 4.3.4 Running model (ps)

```bash
# Check the model while it is running/loading
ollama ps

# Output example
NAME              ID        SIZE    PROCESSOR    UNTIL
llama3.1:latest   abc123    4.7GB   100% GPU     4 minutes from now
qwen2.5:7b        def456    4.9GB   100% GPU     4 minutes from now
```

**Description of output items:**
- **NAME**: Model name
- **SIZE**: Memory usage
- **PROCESSOR**: GPU/CPU usage rate
- **UNTIL**: Time until unloading (unloading after 5 minutes of idle)

### 4.3.5 Copy model (cp)

```bash
# Copy model (save with custom name)
ollama cp llama3.1 my-assistant

# Use
ollama run my-assistant

# confirmation
ollama list
# my-assistant has been added
```

### 4.3.6 Delete model (rm)

```bash
# Delete model
ollama rm mistral:latest

# Delete multiple
ollama rm llama3.1:8b qwen2.5:3b

# Display confirmation prompt
# deleted 'mistral:latest'
```

**⚠️ Note:** Deletion cannot be undone. If you want to use it again, you need to download it again using `ollama pull`.

## 4.4 Practical usage examples

### 4.4.1 Text generation

```bash
# Story generation
ollama run llama3.1 "Write a short story about a robot learning to love."

# Code explanation
ollama run qwen2.5:7b "Explain this Python code:
$(cat script.py)
"

# summary
ollama run llama3.1 "Summarize in 3 sentences:
$(cat article.txt)
"
```

### 4.4.2 Translation

```bash
# English→Japanese
ollama run qwen2.5:14b "Translate to Japanese:
Artificial Intelligence is transforming the world.
"

# Japanese→English
ollama run llama3.1 "Translate to English:
Artificial intelligence is transforming the world.
"

# Multi-language translation script
translate() {
    local text="$1"
    local target="$2"
    ollama run qwen2.5:14b "Translate to $target: $text"
}

translate "Hello World" "Japanese"
translate "Hello World" "Spanish"
translate "Hello World" "French"
```

### 4.4.3 Coding assistance

```bash
# code generation
ollama run codellama:13b "Write a Python function to calculate fibonacci numbers"

# code review
ollama run qwen2.5:14b "Review this code and suggest improvements:
$(cat mycode.py)
"

# Bug fixes
ollama run llama3.1 "Find and fix the bug in this code:
def calculate(x, y):
    return x + x  # Bug here
"

# output
# The bug is that it adds x to itself instead of adding x and y.
# Fixed version:
# def calculate(x, y):
#     return x + y
```

### 4.4.4 Data processing

```bash
# Generate JSON
ollama run llama3.1 --format json "Generate a JSON object with 3 users containing name, age, and email"

# CSV analysis
ollama run qwen2.5:7b "Convert this data to JSON:
$(cat data.csv)
"

# data extraction
ollama run llama3.1 "Extract all email addresses from this text:
$(cat document.txt)
"
```

### 4.4.5 Information extraction and analysis

```bash
# Keyword extraction
ollama run qwen2.5:14b "Extract the main keywords from this article:
$(cat article.txt)
"

# Sentiment analysis
ollama run llama3.1 "Analyze the sentiment (positive/negative/neutral) of this review:
The MS-S1 Max is an amazing machine! The performance is outstanding.
"

# entity recognition
ollama run qwen2.5:7b "Extract all person names, locations, and organizations from:
$(cat news.txt)
"
```

## 4.5 Utilization in shell scripts

### 4.5.1 Basic script example

```bash
#!/bin/bash
# ollama_translate.sh

MODEL="qwen2.5:14b"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <text_to_translate>"
    exit 1
fi

TEXT="$1"
ollama run $MODEL "Translate to Japanese: $TEXT"
```

```bash
# Use
chmod +x ollama_translate.sh
./ollama_translate.sh "Hello, how are you?"
```

### 4.5.2 Batch processing

```bash
#!/bin/bash
# batch_summarize.sh

MODEL="llama3.1"
INPUT_DIR="./articles"
OUTPUT_DIR="./summaries"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.txt; do
    filename=$(basename "$file" .txt)
    echo "Processing: $filename"

    ollama run $MODEL "Summarize this article in 2-3 sentences:
$(cat "$file")
" > "$OUTPUT_DIR/${filename}_summary.txt"

    echo "Saved: $OUTPUT_DIR/${filename}_summary.txt"
done

echo "Batch processing completed!"
```

### 4.5.3 Interactive Menu

```bash
#!/bin/bash
# ollama_menu.sh

while true; do
    echo "==============================="
    echo "  Ollama Assistant Menu"
    echo "==============================="
    echo "1. Chat (Llama 3.1)"
    echo "2. Translate (Qwen 2.5)"
    echo "3. Code Help (CodeLlama)"
    echo "4. Exit"
    echo "==============================="
    read -p "Select option: " choice

    case $choice in
        1)
            ollama run llama3.1
            ;;
        2)
            read -p "Enter text to translate: " text
            ollama run qwen2.5:14b "Translate to Japanese: $text"
            ;;
        3)
            read -p "Describe the code you need: " desc
            ollama run codellama:13b "$desc"
            ;;
        4)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
done
```

### 4.5.4 Script with logging function

```bash
#!/bin/bash
# ollama_with_log.sh

MODEL="qwen2.5:7b"
LOG_DIR="$HOME/ollama_logs"
LOG_FILE="$LOG_DIR/chat_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

echo "Starting Ollama session..."
echo "Log file: $LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while IFS= read -r -p "You: " prompt; do
    [ -z "$prompt" ] && continue
    [ "$prompt" = "exit" ] && break

    echo "You: $prompt" >> "$LOG_FILE"
    echo "Assistant:" | tee -a "$LOG_FILE"

    response=$(ollama run $MODEL "$prompt")
    echo "$response" | tee -a "$LOG_FILE"
    echo "" >> "$LOG_FILE"
done

echo "Session ended. Log saved to: $LOG_FILE"
```

## 4.6 Multi-model utilization

### 4.6.1 Different uses of models

```bash
# Small and high speed model (simple tasks)
FAST_MODEL="qwen2.5:7b"

# Medium/balanced model (general tasks)
BALANCED_MODEL="qwen2.5:14b"

# Large, high-precision models (complex tasks)
POWERFUL_MODEL="qwen2.5:32b"

# Use according to purpose
ollama run $FAST_MODEL "What is 2+2?"
ollama run $BALANCED_MODEL "Explain quantum entanglement"
ollama run $POWERFUL_MODEL "Write a detailed business plan for a startup"
```

### 4.6.2 Utilizing specialized models

```bash
# Coding: CodeLlama
ollama run codellama:13b "Write a sorting algorithm in Python"

# Mathematics: Qwen2.5-Math
ollama pull qwen2.5-math:7b
ollama run qwen2.5-math:7b "Solve: ∫x²dx"

# Japanese language specialization: ELYZA
ollama pull elyza:jp8b
ollama run elyza:jp8b "Tell me about Japanese history"
```

### 4.6.3 Pipeline processing

```bash
# Step 1: Summary
SUMMARY=$(ollama run llama3.1 "Summarize in one sentence: $(cat article.txt)")

# Step 2: Translation
TRANSLATION=$(ollama run qwen2.5:14b "Translate to Japanese: $SUMMARY")

# Step 3: Keyword extraction
KEYWORDS=$(ollama run qwen2.5:7b "Extract 3 keywords from: $TRANSLATION")

echo "Summary: $SUMMARY"
echo "Translation: $TRANSLATION"
echo "Keywords: $KEYWORDS"
```

## 4.7 Performance measurement

### 4.7.1 Execution time measurement

```bash
# Measure with time command
time ollama run llama3.1 "Write a 100-word essay on AI"

# Output example
real    0m8.234s
user    0m0.042s
sys     0m0.028s
```

### 4.7.2 Token Velocity Calculation

```bash
#!/bin/bash
# measure_speed.sh

MODEL="$1"
PROMPT="$2"
NUM_TOKENS=100

START=$(date +%s.%N)
ollama run $MODEL "$PROMPT" > /dev/null
END=$(date +%s.%N)

ELAPSED=$(echo "$END - $START" | bc)
SPEED=$(echo "scale=2; $NUM_TOKENS / $ELAPSED" | bc)

echo "Model: $MODEL"
echo "Time: ${ELAPSED}s"
echo "Speed: ${SPEED} tokens/s"
```

### 4.7.3 Comparing multiple models

```bash
#!/bin/bash
# compare_models.sh

MODELS=("qwen2.5:7b" "qwen2.5:14b" "llama3.1" "mistral")
PROMPT="Explain machine learning in 50 words"

echo "Model Comparison Results:"
echo "========================="

for model in "${MODELS[@]}"; do
    echo -n "Testing $model... "
    START=$(date +%s.%N)
    ollama run $model "$PROMPT" > /dev/null 2>&1
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    echo "Time: ${ELAPSED}s"
done
```

## 4.8 Troubleshooting

### 4.8.1 Command not found

```bash
# error
ollama: command not found

# Solved
echo $PATH # check if /usr/local/bin is included

# if not included
export PATH=/usr/local/bin:$PATH

# Persistence
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
```

### 4.8.2 Model does not start

```bash
# Error example
Error: model 'llama3.1' not found

# Solution 1: Download the model
ollama pull llama3.1

# Solution 2: Check the correct name
ollama list
```

### 4.8.3 Slow response

```bash
# Confirm GPU usage
rocm-smi

# If GPU usage is low, it may be running on CPU
# Check ROCm settings by referring to Chapter 3
```

## 4.9 Summary of this chapter

In this chapter, you learned the following contents.

✅ **Basic commands**
- run, pull, list, show, ps, rm etc.

✅ **Run mode**
- Interactive mode
- One shot execution
- Execute from standard input

✅ **Model management**
- Download, delete, copy
- Information display

✅ **Practical usage**
- Text generation, translation and coding assistance
- Data processing, information extraction

✅ **Scripted**
- Automation with shell scripts
- Batch processing, logging function

In the next chapter, you'll learn how to create custom models using Modelfiles and learn advanced model management techniques.

---

**Go to previous chapter**: [Chapter 3 ROCm settings and AMD GPU optimization](chapter03_rocm_optimization.md)
**Next Chapter**: [Chapter 5 Model Management and Customization](chapter05_model_management.md)
