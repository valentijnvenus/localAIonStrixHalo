# Chapter 6: Complete explanation of inference settings

## 6.1 Basics of inference parameters

### 6.1.1 Understanding the Reasoning Process

LLM reasoning (text generation) is a probabilistic process.

**Basic flow:**

```
1. Prompt input
   ↓
2. Tokenization (dividing sentences into tokens)
   ↓
3. Encoding (context understanding)
   ↓
4. Decoding (predicting the next token)
   ↓
5. Sampling (select from probability distribution)
   ↓
6. Token generation
   ↓
7. Return to 4 (repeat until end condition)
```

**Sampling** is the heart of the inference setup. At each step, the model outputs the probability distribution of the next token.

```
Example: Next token probability distribution for "I am"

"Student": 25%
"Teacher": 18%
"Doctor": 12%
"Engineer": 10%
...others...
```

Inference parameters control how tokens are selected from this probability distribution.

### 6.1.2 Deterministic vs. stochastic generation

**Deterministic generation (Temperature = 0)**
```
Feature: Always select the token with the highest probability
Result: same prompt, always same output
Uses: fact checking, code generation, when consistency is important
```

**Stochastic generation (Temperature > 0)**
```
Features: Randomly select tokens based on probability
Result: Same prompt, different output
Uses: creative writing, brainstorming, diverse responses
```

## 6.2 Detailed explanation of main parameters

### 6.2.1 Temperature

**Definition:** Parameter that controls the "sharpness" of the probability distribution

#### Setting range and effects

```
Setting range: 0.0 ~ 2.0
Default: 0.7
Recommended range: 0.3 to 1.0
```

**Relationship between numbers and effects:**

**Temperature = 0.0 (deterministic)**
```
Probability distribution:
"Student": 100% ← Must be selected
"Teacher": 0%
"Doctor": 0%

Features:
✓ Completely predictable
✓ Highly consistent
✓ Zero creativity

Usage:
- Answers to math problems
- Code generation
- Enumeration of facts
- Translation (focus on accuracy)
```

**Temperature = 0.3 (low temperature/conservative)**
```
Probability distribution:
"Student": 60%
"Teacher": 30%
"Doctor": 8%
Others: 2%

Features:
✓ Highly predictable
✓ SAFE CHOICE
✓ Minor variations

Usage:
- Business documents
- Technical documentation
- FAQ answers
- Customer support
```

**Temperature = 0.7 (Standard/Balanced)**
```
Probability distribution:
"Student": 32%
"Teacher": 24%
"Doctor": 16%
"Engineer": 13%
Others: 15%

Features:
✓ Good balance
✓ Moderate creativity
✓ Natural conversation

Usage:
- Daily conversation
- Blog article
- Email reply
- General chat ← Most recommended
```

**Temperature = 1.0 (high temperature/creative)**
```
Probability distribution:
"Student": 25%
"Teacher": 20%
"Doctor": 15%
"Engineer": 12%
"Artist": 10%
Others: 18%

Features:
✓ Creative
✓ Diverse expressions
✓ Difficult to predict

Usage:
- Writing novels and stories
- Brainstorming
- Poetry generation
- Idea generation
```

**Temperature = 1.5-2.0 (very high temperature/experimental)**
```
Probability distribution: almost uniform

Features:
⚠️ Very random
⚠️ Lack of consistency
⚠️ Possibility of nonsense output

Usage:
- Experimental creation
- If you want randomness
Note: Not recommended for normal use
```

**💡 TIP**: For MS-S1 Max, Temperature 0.7 is optimal for most applications.

### 6.2.2 Top P（Nucleus Sampling）

**Definition:** Select from a set of tokens until the cumulative probability reaches P

#### Setting range and effects

```
Setting range: 0.0 ~ 1.0
Default: 0.95
Recommended range: 0.85 to 1.0
```

**Working principle:**

```
Probability distribution (descending order):
"Student": 25% Cumulative: 25% ← Top If P = 0.25, that's it
"Teacher": 18% Cumulative: 43% ← Top If P = 0.43, that's it
"Doctor": 12% Cumulative: 55%
"Engineer": 10% Cumulative: 65%
"Artist": 8% Cumulative: 73%
"Chef": 6% Cumulative: 79%
  ...

For Top P = 0.95:
Select from tokens up to 95% cumulative
→ Approximately 20-30 tokens are candidates
```

**Numbers and effects:**

**Top P = 0.5 (very restrictive)**
```
effect:
- Only tokens with top 50% probability are considered
- very conservative
- Highly predictable

Recommended use:
- Factual answers
- Professional content
```

**Top P = 0.75 (slightly restrictive)**
```
effect:
- Select from top 75% tokens
- Balanced output
- Moderate diversity

Recommended use:
- Business documents
- Technical documentation
```

**Top P = 0.95 (typical)**
```
effect:
- Select from top 95% tokens
- natural diversity
- Also consider low probability tokens

Recommended use:
- Daily conversation ← Most recommended
- creative writing
```

**Top P = 1.0 (unlimited)**
```
effect:
- All tokens are candidates
- maximum diversity
- Rare tokens can also be selected

Recommended use:
- Experimental creation
- Maximum creativity
```

**💡 TIP**: Top P = 0.95 is optimal in most cases.

### 6.2.3 Top K

**Definition:** Select only from the top K tokens with the highest probability

#### Setting range and effects

```
Setting range: 0 to 100 (normal)
Default: 40
Recommended range: 20-60
Special values: 0 = disabled (do not use Top K)
```

**Working principle:**

```
Probability distribution:
1. "Student": 25%
2. "Teacher": 18%
3. "Doctor": 12%
4. "Engineer": 10%
5. "Artist": 8%
... (continued)

If Top K = 3:
→ Select only from the top 3 (student, teacher, doctor)

If Top K = 40:
→ Select from top 40 tokens
```

**Top P vs Top K:**

```
Top P（Nucleus Sampling）:
✓ Dynamic number of candidates (depending on probability distribution)
✓ Adapt to context
✓ More natural output
Recommended: ✅

Top K:
✓ Fixed number of candidates
✓ Predictable
✓ Simple to implement
Recommended: Use as a supplement
```

**Recommended settings:**

```
Standard settings:
  Top P = 0.95
  Top K = 40

Conservative settings:
  Top P = 0.85
  Top K = 20

Creative settings:
  Top P = 1.0
  Top K = 60
```

### 6.2.4 Repeat Penalty

**Definition:** Penalize reappearance of already generated tokens

#### Setting range and effects

```
Setting range: 1.0 ~ 1.5
Default: 1.1
Recommended range: 1.05 to 1.15
Special value: 1.0 = no penalty
```

**Working principle:**

```
Generated text: "I'm a student. I..."

Next token prediction (before penalty):
"Student": 30%
"Teacher": 20%
"Daily": 15%

After applying Repeat Penalty = 1.1:
"Student": 30% / 1.1 = 27.3% ← Penalty
"I": 25% / 1.1 = 22.7% ← Penalty
"Ha": 22% / 1.1 = 20% ← Penalty
"Teacher": 20% (no change)
"Daily": 15% (no change)

After renormalization, select
```

**Numbers and effects:**

**Repeat Penalty = 1.0 (no penalty)**
```
Effect: Does not prevent repetition
result:
⚠️ Repeating the same phrase
⚠️ "And, and, and..."
Recommended use: Not normally used
```

**Repeat Penalty = 1.05 (minor)**
```
Effect: Minor penalty
Features:
✓ Allow natural repetition
✓ Prevent excessive repetition
Recommended use:
- Poetry, lyrics (with intentional repetition)
- Natural conversation
```

**Repeat Penalty = 1.1 (Standard)**
```
Effect: Moderate penalty
Features:
✓ Good balance
✓ Prevent unnatural repetition
✓ Maintain vocabulary diversity
Recommended use:
- Daily conversation ← Most recommended
- Sentence generation
- Most uses
```

**Repeat Penalty = 1.15-1.2 (strong)**
```
Effect: Strong penalty
Features:
✓ Almost prevents repetition
⚠️ May look a little unnatural
Recommended use:
- Long sentence generation
- Create list (avoid duplication)
```

**Repeat Penalty = 1.3+ (very strong)**
```
Effect: Very strong penalty
result:
⚠️ Unnatural vocabulary selection
⚠️ Words that are out of context
Recommended use: Not normally used
```

### 6.2.5 Context Length

**Definition:** Number of tokens that the model can process at once

#### Settings and tradeoffs

```
Setting range: 512 ~ model maximum value (8K, 32K, 128K, etc.)
Default: model dependent (usually 4K-8K)
Recommended settings: Adjust according to usage
```

**Maximum context by model:**

| Model | Maximum Context | Recommended Settings (MS-S1 Max) |
|--------|----------------|----------------------|
| Qwen2.5 7B | 32K (32768) | 16K-32K |
| Qwen2.5 14B | 32K | 16K-32K |
| Qwen2.5 32B | 32K | 16K-32K |
| Qwen2.5 72B | 128K (131072) | 32K-64K |
| Llama 3.1 8B | 128K | 32K-64K |
| Llama 3.1 70B | 128K | 16K-32K |

**Context length and memory usage:**

```
Basic formula:
Additional memory ≈ (context length / 1024) × (number of parameters / 1B) × 0.15 GB

Example: Qwen2.5 7B, 32K context
Additional memory ≈ (32768 / 1024) × (7.62 / 1) × 0.15
          ≈ 32 × 7.62 × 0.15
          ≈ 36.6 GB

Model body: 4.8GB
Context: 36.6GB (when set to 32K)
Total: Approximately 41GB
```

**Context length selection:**

**2K-4K (short)**
```
Usage:
- Short question and answer
- Simple chat
- When fast response is required

Memory: minimum
Speed: Fastest
```

**8K-16K (moderate)**
```
Usage:
- normal conversation
- Moderate document processing
- Balanced type ← Recommended

Memory: Moderate
Speed: Fast
```

**32K-64K (long)**
```
Usage:
- Long text summary
- Compare multiple documents
- Maintain long conversation history

Memory: large
Speed: medium speed
```

**128K (very long)**
```
Usage:
- Whole book analysis
- Processing of extremely long sentences
- Professional use

Memory: very large
Speed: Slightly slow
Constraints: Difficult with 70B model even with 128GB memory
```

**💡 TIP**: For MS-S1 Max, 16K-32K is a practical balance point.

### 6.2.6 Max Tokens (Maximum number of generated tokens)

**Definition:** Maximum number of tokens to generate in one response

```
Setting range: 1 to 32768 (less than or equal to context length)
Default: 2048
Recommended range: 512-4096
```

**Recommended value by application:**

```
Short response (chat): 512-1024
Medium response (description): 1024-2048
Long response (sentence generation): 2048-4096
Very long (article writing): 4096-8192
```

**⚠️ Note**: Even if you set a large value, if the model generates an exit condition (EOS), it will stop there.

### 6.2.7 Other advanced parameters

#### Frequency Penalty

```
Setting range: 0.0 ~ 2.0
Default: 0.0
Recommended: 0.0~0.3

effect:
- Penalty based on the number of times the token has already appeared
- More control over Repeat Penalty
- 0.0 = no penalty
```

#### Presence Penalty

```
Setting range: 0.0 ~ 2.0
Default: 0.0
Recommended: 0.0~0.3

effect:
- Penalty on whether the token has already appeared
- Facilitate transition to new topics
- 0.0 = no penalty
```

#### Mirostat

```
Settings: 0, 1, 2
Default: 0 (disabled)

Mirostat 0: Disabled
Mirostat 1: Dynamic perplexity control
Mirostat 2: Improved version

effect:
- Dynamically adjust output consistency
- Overwrite Temperature, Top P, Top K
- Experimental features

Recommended: Usually disabled (0)
```

## 6.3 Utilizing presets

### 6.3.1 Default Preset

LM Studio provides presets for different purposes.

#### Precise

```
Temperature: 0.3
Top P: 0.85
Top K: 20
Repeat Penalty: 1.1
Context: 8K

Usage:
✓ Fact checking
✓ Code generation
✓ Technical documentation
✓ Translation (focus on accuracy)
```

#### Balanced

```
Temperature: 0.7
Top P: 0.95
Top K: 40
Repeat Penalty: 1.1
Context: 16K

Usage:
✓ Daily conversation ← Most recommended
✓ Generic chat
✓ Create email
✓ Blog article
```

#### Creative

```
Temperature: 1.0
Top P: 1.0
Top K: 60
Repeat Penalty: 1.05
Context: 16K

Usage:
✓ Novel writing
✓ Brainstorming
✓ Poetry generation
✓ Idea generation
```

### 6.3.2 Creating a custom preset

**Preset creation procedure**

1. **Adjust parameters in Chat tab**
   ```
Model selection screen → ⚙️ Settings
Adjust each parameter
   ```

2. **Save as preset**
   ```
Settings screen bottom → “Save as Preset”
Enter a name (e.g. "My Coding Assistant")
Add description (optional)
Click "Save"
   ```

3. **Apply preset**
   ```
Chat tab → Preset dropdown → select "My Coding Assistant"
   ```

**Recommended custom preset example:**

#### Japanese chat optimization

```yaml
Name: Japanese Chat Optimized
Temperature: 0.8
Top P: 0.95
Top K: 45
Repeat Penalty: 1.12
Context Length: 16384
Max Tokens: 2048
Recommended model: Qwen2.5 7B/14B
```

#### Coding Assistant

```yaml
Name: Coding Assistant
Temperature: 0.2
Top P: 0.9
Top K: 30
Repeat Penalty: 1.05
Context Length: 8192
Max Tokens: 4096
Recommended model: DeepSeek-Coder V2
```

#### Long text generation

```yaml
Name: Long Form Writing
Temperature: 0.7
Top P: 0.92
Top K: 40
Repeat Penalty: 1.15
Context Length: 32768
Max Tokens: 8192
Recommended model: Qwen2.5 32B
```

### 6.3.3 Share and import presets

**Export preset**

```
Settings → Presets → Right-click on the target preset
→ Export → Specify file name (.json)
```

**Import preset**

```
Settings → Presets → Import
→ Select .json file
```

## 6.4 Practical parameter tuning

### 6.4.1 Recommended settings for each application

#### Case study 1: Technical documentation creation

```
Purpose: Accurate and professional technical documentation
Recommended settings:
  Model: Qwen2.5 14B Q4_K_M
  Temperature: 0.4
  Top P: 0.88
  Top K: 25
  Repeat Penalty: 1.12
  Context: 16K
  Max Tokens: 4096

reason:
- Low Temperature → Focus on facts
- Low Top P/K → Accurate technical terms
- Slightly higher Repeat Penalty → Diverse expressions
```

#### Case Study 2: Creative Writing

```
Purpose: Creative novel writing
Recommended settings:
  Model: Qwen2.5 32B Q4_K_M
  Temperature: 0.9
  Top P: 0.98
  Top K: 55
  Repeat Penalty: 1.08
  Context: 32K
  Max Tokens: 8192

reason:
- High Temperature → Creativity
- High Top P/K → Diverse vocabulary
- Low Repeat Penalty → Natural writing style
- Long context → story consistency
```

#### Case Study 3: Customer Support

```
Purpose: Consistent customer care
Recommended settings:
  Model: Qwen2.5 7B Q4_K_M
  Temperature: 0.5
  Top P: 0.90
  Top K: 30
  Repeat Penalty: 1.10
  Context: 8K
  Max Tokens: 1024

reason:
- Medium-low Temperature → Safe response
- Moderate Top P/K → Moderate flexibility
- Short Max Tokens → Concise response
```

### 6.4.2 Troubleshooting

#### Problem 1: Too much repetition

```
Symptoms:
"And, and, and..."
Continuation of the same phrase

Solution:
✓ Increase Repeat Penalty: 1.1 → 1.15
✓ Enable Frequency Penalty: 0.2
✓ Increase Temperature a little: 0.7 → 0.8
```

#### Problem 2: Inconsistent/disjointed

```
Symptoms:
The topic keeps changing
Responses taken out of context

Solution:
✓ Lower Temperature: 0.9 → 0.6
✓ Lower Top P: 0.98 → 0.92
✓ Lower Top K: 60 → 35
```

#### Problem 3: Lack of creativity/boring

```
Symptoms:
similar response every time
too predictable

Solution:
✓ Increase Temperature: 0.5 → 0.8
✓ Increase Top P: 0.85 → 0.95
✓ Lower Repeat Penalty: 1.15 → 1.05
```

#### Problem 4: Stopping midway

```
Symptoms:
Sentence ends midway
incomplete response

Solution:
✓ Increase Max Tokens: 1024 → 2048
✓ Check Context Length (is it sufficient?)
✓ Reload model
```

## 6.5 Relationship with performance

### 6.5.1 Parameters and speed

**Parameters that affect speed:**

```
Context Length: Large ← Most impactful
8K → 16K: Speed ​​approximately 15% lower, memory approximately double
16K → 32K: Speed ​​approximately 15% lower, memory approximately double

Max Tokens: Medium impact
1024 → 4096: Generation time 4 times (proportional to the number of tokens)

Temperature, Top P, Top K: Small
Sampling calculation overhead is minor
```

**MS-S1 Max optimization settings:**

```
Fast priority:
  Context: 8K
  Max Tokens: 1024
Estimated speed: 35-45 t/s (7B model)

balance:
  Context: 16K
  Max Tokens: 2048
Estimated speed: 30-40 t/s (7B model)

Quality first:
  Context: 32K
  Max Tokens: 4096
Estimated speed: 25-35 t/s (7B model)
```

### 6.5.2 Relationship between memory and context length

**Memory usage calculation table (Qwen2.5 7B Q4_K_M):**

| Context Length | Model | Context | Total |
|---------------|--------|-------------|------|
| 4K | 4.8GB | 9GB | 13.8GB |
| 8K | 4.8GB | 18GB | 22.8GB |
| 16K | 4.8GB | 36GB | 40.8GB |
| 32K | 4.8GB | 72GB | 76.8GB |

**💡 TIP**: Even with MS-S1 Max's 128GB memory, using 32K on the 7B model will consume approximately 77GB. Leave room for other applications.

## 6.6 Summary of this chapter

In this chapter, you learned more about inference settings.

✅ **Main parameters**
- Temperature: 0.7 (Standard), 0.3 (Precision), 1.0 (Creative)
- Top P: 0.95 (recommended)
- Top K: 40 (recommended)
- Repeat Penalty: 1.1 (standard)
- Context Length: 16K (recommended, adjust according to usage)

✅ **Use presets**
- Precise、Balanced、Creative
- Create custom presets

✅ **Optimization by use**
- Technical Documentation: Low Temperature, Short Context
- Creation: High Temperature, Long Context
- Chat: Balanced

✅ **Troubleshooting**
- Repeat → Increase Repeat Penalty
- Incoherent → Temperature decrease
- Lack of creativity → Increased Temperature

In the next chapter, we will learn about optimization settings specific to MS-S1 Max.

---

**Go to previous chapter**: [Chapter 5 Model download and management](chapter05_model_management.md)
**Next chapter**: [Chapter 7 Optimization settings for MS-S1 Max](chapter07_optimization.md)
