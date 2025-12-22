# Chapter 05: Feature compatibility and limitations

**📖 Purpose of this chapter**

In this chapter, we will help you understand the difference between Local LLM (Qwen3 Coder) and Claude API so that you can decide if it can be used successfully for your purpose.

**🎯 Points for readers to decide**
1. Does the function you want to use work?
2. Is it possible to work around limited functionality?
3. Is local LLM sufficient or do you need Claude API?

## 5.1 Difference between local LLM and Claude API

### 5.1.1 Architectural differences

**Claude API (main family)**
```
Claude Code → Anthropic API → Claude 3.5 Sonnet
               ↓
- Dedicated protocol
- Function Calling
- Vision (image understanding)
- Long context (200K tokens)
- Streaming
```

**Local LLM (Structure of this book, latest as of November 2025)**
```
Aider → LiteLLM Proxy → Ollama → Qwen3 Coder 30B Q8_0
         ↓
- OpenAI compatible protocols
- limited function calls
- text only
- Context (256K tokens, expandable up to 1M)
- Streaming
- MS-S1 Max: 96GB VRAM supported
```

### 5.1.2 Feature comparison table

| Features | Claude API | Local LLM (Qwen3 Coder 30B) | How to deal with it | Impact on you |
|------|-----------|------------------------------|---------|---------------|
| Chat completion | ✅ | ✅ | Fully compatible | **No problems**・No problems with normal coding support |
| Streaming | ✅ | ✅ | Fully compatible | **No problems**・Real-time display possible |
| Function call | ✅ | ⚠️ Limited | Replacement with a prompt | **Almost no problem**・Can be handled by devising a prompt |
| Vision (image) | ✅ | ❌ | Uses LLaVA (separately) | **Depends on usage**/Separate model if screenshot analysis is required |
| Long context | ✅ 200K | ✅ 256K (1M expandable) | Local is advantageous | **Local is advantageous**・Large-scale file processing possible |
| Multilingual support | ✅ | ✅ | No problems | **No problems**・High quality in both Japanese and English |
| Code generation | ✅ | ✅ | High quality with Qwen3 | **No problems**・Compatible with coding specialized model |
| Japanese | ✅ | ✅ | High quality | **No problems**・Qwen3 is good at Japanese |
| Response speed | ⚠️ Network | ✅ 22 tokens/s | Local is faster | **Local advantage**・Reduced waiting time |
| Privacy | ❌ External transmission | ✅ Completely local | Local is advantageous | **Local advantage**・Secret code security |
| Cost | ❌ Paid | ✅ Free | Local is advantageous | **Local is advantageous**・Unlimited usage |
| VRAM requirements | ❌ N/A | ✅ 32GB (MS-S1: 96GB possible) | MS-S1 is sufficient | **Initial investment only**・MS-S1 Max recommended |

**📊 How to read this table**

**✅ Fully compatible**: Ready to use. Please use it without worrying about anything.

**⚠️ Restrictions**: Can be used, but requires some modification. This chapter explains how to avoid it.

**❌ Not supported**: The function cannot be used. Please consider alternatives.

**🤔 Check your usage**

Use the checklist below to determine if a local LLM is right for you:

```
□ Main use is code generation/editing
□ Mainly text-based question and answer
□ No image/screenshot analysis required
□ I don't want to send confidential information to outside parties
□ I want to reduce API costs
□ I want to use it in an offline environment

→ Check 3 or more: Local LLM recommended
→ Check 1-2: Local LLM is sufficient
→ 0: Claude API consideration
```

**❓ Frequently asked questions**

**Q: Can I use all the features of Claude Code? **
A: No. Some functions, such as vision functions, cannot be used. However, the main features of Coding Assistance (code generation, editing, and review) work fine.

**Q: What features do you have trouble not being able to use? **
A: The main limitation is that **image understanding** cannot be used. A separate LLaVA model is required to find bugs from screenshots, generate code from UI designs, etc.

**Q: I can't decide if it can be used for my purpose**
A: **Please tell us the purpose**:
- Python script creation → ✅ No problems
- Web application development → ✅ No problems
- Data analysis → ✅ No problems
- From UI design to implementation → ⚠️ If image understanding is required, use a different model
- Screenshot analysis → ❌ Requires LLaVA etc.

## 5.2 Functions that work

### 5.2.1 Fully working functionality

**💡 Criteria**: The following features work the same as the Claude API. You can use it without worrying about anything.

**1. Text-based code generation**

```
> Create a Python function to calculate prime numbers

# ✅ Fully working
```

**To try it in your environment**:
```bash
# start Aider
aider

# prompt input
> Create a Python function to calculate prime numbers
```

**Operation example**

```python
def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(limit):
    """Generate prime numbers up to limit."""
    return [n for n in range(2, limit + 1) if is_prime(n)]

# Example usage
print(get_primes(100))
```

**💬 What you can do**:
- Generation of new functions/classes
- Implementation of the algorithm
- Creating boilerplate code
- Definition of data structures

**2. File editing**

```
> /add main.py
> Add error handling to this function

# ✅ Fully working
# Aider reads the file and applies edits
```

**To try it in your environment**:
```bash
aider
> /add your_file.py
> Add try-except error handling to all functions
```

**💬 What you can do**:
- Modification of existing code
- Refactoring
- Bug fixes
- Added functionality

**3. Code review**

```
> /add complex_function.py
> Review this code and suggest improvements

# ✅ Fully working
```

**Output example**
```
Here are my suggestions:

1. **Type Hints**: Add type annotations for better code clarity
2. **Error Handling**: Add try-except blocks for file operations
3. **Docstrings**: Add comprehensive documentation
4. **Performance**: Use list comprehension instead of loops
5. **Naming**: Rename variable 'x' to 'result' for clarity

Would you like me to apply these changes?
```

**4. Refactoring**

```
> Refactor this code to use async/await

# ✅ Behavior (depends on model capabilities)
```

**5. Test generation**

```
> Generate pytest tests for this module

# ✅ Fully working
```

**6. Document generation**

```
> Create a README.md with usage examples

# ✅ Fully working
```

**7. Bug fixes**

```
> Fix the IndexError in line 45

# ✅ Operation (accuracy depends on model)
```

### 5.2.2 Partially working features

**💡 Judgment Criteria**: The following functions can be used, but some modifications are required. **Please check if it is necessary for your application**.

**1. Function Calling**

Function Calling in the Claude API has limited functionality.

**🤔 Does it affect you? **
```
□ I want to automatically generate a tool that calls the API
□ I want to automatically execute database queries
□ I want to link with external services

→ Checked: Workaround (prompt improvement) required
→ Unchecked: No effect/unnecessary for normal coding
```

**Workaround: Prompt Engineering**

```
# ❌ Claude API style (doesn't work)
tools = [
    {
        "name": "search_file",
        "description": "Search for a file",
        "parameters": {...}
    }
]

# ✅ Alternative with prompt (works)
> I need to search for files containing "TODO".
> Please suggest a bash command to do this.

# Model suggests:
# grep -r "TODO" ./src/
```

**2. Multi-step tasks**

The original Claude Code can automatically break down complex tasks, but local LLM requires explicit instructions.

**🤔 Does it affect you? **
```
□ I want to implement large-scale functions all at once
□ I would like to request a complex architecture change.
□ I want tasks to be automatically broken down

→ Checked: Step-by-step instructions required
→ Not checked: No impact/No problem for small tasks
```

**Workaround: Step-by-step instructions**

```
# ⚠️ Ambiguous instructions (unstable operation)
> Create a web app with user authentication

# ✅ Step by step (stable operation)
> Step 1: Create a FastAPI server with a /login endpoint
# ... after completion ...
> Step 2: Add JWT token generation
# ... after completion ...
> Step 3: Create middleware for authentication
```

**💡 Best Practices**:
- Request one feature with one prompt
- Confirm completion before proceeding to the next step
- Divide large tasks into 3-5 steps

**3. Understanding large codebases**

Qwen3 Coder 30B Q8_0 has a context of 256K tokens (native) and is scalable up to 1M. MS-S1 Max's 96GB VRAM can handle even large projects.

**Recommended: Add files selectively (for efficient processing)**

```bash
# ⚠️ Add entire project (possible but inefficient)
> /add src/**/*.py # Possible within 256K

# ✅ Add only related files (recommended/efficient)
> /add src/auth.py src/models.py src/utils.py

# ✅ Taking advantage of MS-S1 Max's large capacity VRAM
> /add src/module1/ src/module2/ # Comfortable up to a total of 100-200K tokens
```

## 5.3 Features that don't work

### 5.3.1 Functionality that does not work completely

**💡 Judgment Criteria**: The following functions cannot be used. **Please check if it is necessary for your application**.

**1. Image understanding (vision)**

```
# ❌ Doesn't work
> Analyze this screenshot and find the bug
> (screenshot.png)

# Qwen3 Coder 30B Q8_0 only supports text
```

**🤔 Does it affect you? **
```
□ I want to find bugs from screenshots
□ I want to generate code from UI design images
□ I want to analyze charts and graphs
□ I want to convert handwritten notes into codes

→ Checked: LLaVA model required (separate setup)
→ Not checked: No effect/text only is sufficient
```

**💡 Not required for most coding uses**:
Normal programming (code generation, review, refactoring) does not use image understanding. If screenshot analysis is not required, you can ignore this limitation.

**Workaround: Use LLaVA model (only if necessary)**

```bash
# Download LLaVA model separately
ollama pull llava:13b

# Analyze images with Python script
python3 << EOF
import ollama

response = ollama.chat(
    model='llava:13b',
    messages=[{
        'role': 'user',
        'content': 'Analyze this code screenshot',
        'images': ['screenshot.png']
    }]
)
print(response['message']['content'])
EOF
```

**2. Claude-specific prompt format**

```
# ❌ Claude API special tag (doesn't work)
<thinking>...</thinking>
<answer>...</answer>

# ✅ Standard prompt
> Think step by step and provide your answer.
```

**3. Web search integration**

```
# ❌ Doesn't work
> Search the web for latest Python best practices

# Ollama only works offline
```

**Workaround: Provide information manually**

```
> Here's the latest info from Python.org: [Paste]
> Based on this, suggest improvements to my code
```

### 5.3.2 Features with poor performance

**1. Very complex reasoning**

Claude 3.5 Sonnet is good at advanced inference, but local models have limitations.

**Example: Mathematical proof**

```
# ⚠️ Local LLM has low accuracy
> Prove that the sum of two even numbers is always even

# Improved by breaking it down into simpler tasks
> Explain what an even number is
> Show examples of adding two even numbers
> Generalize the pattern
```

**2. Understanding and generating long sentences**

The 32K token limit limits processing of very long sentences.

```
# ⚠️ Restrictions apply
> Summarize this 50-page document

# ✅ Chunking
> Summarize pages 1-10
> Summarize pages 11-20
> ...
> Combine all summaries
```

## 5.4 How to deal with limitations

### 5.4.1 Avoiding context restrictions

**Method 1: Add only related files**

```bash
# Identify related files from Git history
git log --follow --oneline -- src/bug_file.py | head -10

# Add only related files
aider src/bug_file.py src/related_module.py
```

**Method 2: Clearing the context**

```
> /clear

# clear the context and start a new task
```

**Method 3: Use summaries**

```
> /add large_file.py
> Summarize the main functions in this file

# Note the summary and drop the details file
> /drop large_file.py

# Work from the summary
```

### 5.4.2 Improved accuracy

**Method 1: Few-Shot Prompting**

```
> Here are two examples of the coding style I want:
>
> Example 1:
> [Paste code example 1]
>
> Example 2:
> [Paste code example 2]
>
> Now, refactor this function to match this style:
> [Paste code to be refactored]
```

**Method 2: Step-by-step instructions**

```
# ❌ Ambiguous
> Optimize this database query

# ✅ Specific
> Step 1: Add indexes to frequently queried columns
> Step 2: Use query explain to identify bottlenecks
> Step 3: Rewrite subqueries as joins
```

**Method 3: Utilize code reviews**

```
> /add generated_code.py

> Review this code I just generated:
> 1. Check for edge cases
> 2. Verify error handling
> 3. Suggest performance improvements
```

### 5.4.3 How to use multiple models

Define multiple models in LiteLLM settings and switch between them depending on the purpose.

```yaml
# config.yaml (latest November 2025, Qwen3 Coder configuration)
model_list:
# High-speed task (14B/256K context)
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: ollama/qwen3-coder:14b
      num_ctx: 262144

# Normal/high quality task (30B Q8_0/256K context)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      num_ctx: 262144

# Highest quality task (30B Q8_0・256K context)
  - model_name: gpt-4
    litellm_params:
      model: ollama/qwen3-coder:30b-a3b-q8_0
      num_ctx: 262144

# Fastest task (7B/256K context)
  - model_name: claude-3-haiku-20240307
    litellm_params:
      model: ollama/qwen3-coder:7b
      num_ctx: 262144
```

**Usage example**

```
# Draft with fast model
> /model gpt-3.5-turbo
> Draft a function to parse JSON

# Review with high quality model
> /model gpt-4
> Review and improve this function
```

## 5.5 Best Practices

### 5.5.1 Effective prompts

**❌ Bad example**

```
> Fix this
```

**✅ Good example**

```
> This function throws a KeyError when the 'name' field is missing from the input dictionary.
> Add error handling to return None instead of crashing.
> Also add type hints and a docstring.
```

### 5.5.2 Step-by-step approach

**❌ All at once**

```
> Create a complete authentication system with OAuth, JWT, password reset, email verification, and RBAC
```

**✅ Step by step**

```
> Step 1: Create a User model with SQLAlchemy
# confirm completion
> Step 2: Add password hashing with bcrypt
# confirm completion
> Step 3: Implement JWT token generation
# ...
```

### 5.5.3 Context Management

**Regular cleanup**

```
# Clear every 10-15 conversations
> /clear

# or start a new session
> /exit
aider
```

**Grouping related files**

```bash
# Authentication related
aider src/auth.py src/models/user.py src/utils/jwt.py

# API related
aider src/api.py src/routes/ src/schemas/
```

## 5.6 Troubleshooting

### 5.6.1 Poor response quality

**Problem**: The quality of the generated code is lower than expected.

**Solution**

1. **Change model**
```
> /model gpt-4 # switch to larger model
```

2. **Improved prompts**
```
> You are an expert Python developer.
> Write production-ready code with:
> - Type hints
> - Error handling
> - Comprehensive docstrings
> - Unit tests
>
> Now, create a function to...
```

3. **Few-Shot Learning**
```
> Here's an example of the code quality I expect:
> [Paste good code example]
>
> Now, generate similar code for...
```

### 5.6.2 Context Exceeded

**Error**: "Context length exceeded"

**Solution**

```
# check context
> /tokens

# Delete unnecessary files
> /drop unnecessary_file.py

# or clear context
> /clear

# Add only the minimum necessary files
> /add essential_file.py
```

### 5.6.3 Generated code does not work

**Problem**: There is a bug in the generated code

**Solution**

1. **Test run**
```
> /run python3 generated_code.py

# Feedback error message
> The code fails with this error: [Paste error message]
> Please fix it.
```

2. **Step-by-step testing**
```
> Generate the code without running it first
# check code
> Now let's test the function step by step
```

3. **Code Review**
```
> Review the generated code for potential bugs
```

## 5.7 Summary

In this chapter, you learned about the limitations of local LLM and how to work around it.

**Important points**

✅ **Function that works**
- Text-based code generation and editing
- Refactoring
- Test generation
- Code review

⚠️ **Restricted features**
- Function call → substitute with prompt
- Image understanding → Supported by LLaVA
- Long context → Chunking

❌ **Features that do not work**
- Claude specific tags
- Web search

**How ​​to deal with it**
1. Clear and specific prompts
2. Step-by-step approach
3. Proper context management
4. How to use multiple models

**Next steps**
In the next chapter, you will learn about the comparison of coding-specific models and the best choice for MS-S1 Max.

**Verification Checklist**
- [ ] Understand the limitations of local LLM
- [ ] Workarounds can be implemented
- [ ] Write effective prompts
- [ ] Can manage context

Once you have checked everything, move on to Chapter 06!
