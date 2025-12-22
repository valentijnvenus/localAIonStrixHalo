# Chapter 8: Practical usage

## 8.1 Utilizing the chat interface

### 8.1.1 Basic dialogue

LM Studio's chat interface offers a similar usability to ChatGPT.

#### How to write effective prompts

**Principle 1: Be clear and specific**

```
❌ Bad example:
"Tell me about programming"

✅ Good example:
"How to save dictionary data to a JSON format file in Python.
Please let me know with sample code. "
```

**Principle 2: Provide context**

```
❌ Bad example:
"How can we improve this?"

✅ Good example:
"I would like to improve the execution speed of the following Python function.
It becomes slow when processing large amounts of data (more than 100,000 rows).
Please suggest optimization.

```python
def process_data(data_list):
    result = []
    for item in data_list:
        if item > 100:
            result.append(item * 2)
    return result
```
"
```

**Principle 3: Ask questions in stages**

```
Conversation flow:

1. "I want to learn web scraping with Python.
Please tell me the steps for beginners. "

2. "Please tell me more about Beautiful Soup."

3. Select elements with a specific CSS class in "Beautiful Soup"
Please show me the sample code to extract. "
```

### 8.1.2 Utilizing system prompts

**System prompts** define the AI's behavior and role.

#### How to access

```
Chat screen → “⚙️” icon at the top
→ System Prompt section
```

#### Examples of effective system prompts

**Example 1: Programming Assistant**

```
You are an experienced software engineer.
Please respond according to the following rules:

1. Provide a complete example where your code always works
2. Explain each step with comments
3. Also mention edge cases and error handling
4. If there are multiple solutions, explain the pros and cons of each.
5. Focus on code efficiency and best practices

Language: Japanese
Code comments: Japanese
```

**Example 2: Writing Assistant**

```
You are a professional writer.
Please respond with the following characteristics:

1. Write clear and readable sentences
2. Use proper paragraphing and headings
3. Explain technical terms as necessary
4. Explain with specific examples
5. Use desu-masu style

Target audience: General business people
Sentence length: Adjust depending on question (default: medium)
```

**Example 3: Japanese learning assistant**

```
You are a Japanese language tutor helping English speakers learn Japanese.

Rules:
1. Explain grammar points in English
2. Provide example sentences in Japanese with romaji and English translations
3. Point out common mistakes learners make
4. Give cultural context when relevant
5. Be encouraging and patient

Format:
Japanese sentence
Romaji
English translation
Grammar explanation
```

### 8.1.3 Managing multiturn interactions

**Use of conversation history**

LM Studio automatically maintains conversation history.

```
Effective usage:

1. Provide sufficient background information in the first question
2. Use references such as “it” and “said” in subsequent questions.
3. Be clear when changing topics

example:
User: "Please tell me about list comprehensions in Python."
AI: [Detailed explanation]
User: "How can I use that to create a list of even numbers from 1 to 100?"
AI: [Answer using list comprehension]
User: "So, can you tell me more about dictionary comprehensions?"
```

**Conversation reset timing**

```
Recommended reset case:
✓ Completely changes topic
✓ After long time use (memory release)
✓ Model response quality has decreased
✓ Approaching the upper limit of the context length

method:
Top right of Chat screen → “New Chat” button
```

## 8.2 Practical use cases

### 8.2.1 Text creation/editing

#### Create a blog post

**Step 1: Outline generation**

```
prompt:
"With the theme of ``Effective exercise habits that you can do at home''
I want to write a blog article.
Please suggest an article outline (heading structure).
The target audience is exercise beginners, and the article length is assumed to be around 2,000 characters. "
```

**Step 2: Writing each section**

```
prompt:
``Please refer to the ``2. Benefits of home exercise'' section of the outline above.
Please write in 300-400 characters. Please include 2-3 specific examples. "
```

**Step 3: Elaboration**

```
prompt:
"Please make the following sentences more catchy and readable:

[Paste original text]
"
```

#### Create email

**Business email**

```
prompt:
"Please compose a business email with the following content.

Address: Mr. Tanaka, our business partner
Purpose: Request to change next week's meeting schedule
Reason: Due to a sudden business trip.
Proposed date and time: Two candidate dates and times are suggested.
Tone: Polite and Professional"
```

### 8.2.2 Code generation and review

#### Code generation

**Python script generation**

```
prompt:
"Create a Python script that meets the following requirements:

Requirements:
1. Load CSV file
2. Extract rows where the value of a specific column ('age') is 30 or more
3. Save results to new CSV file
4. Include error handling
5. Explain each step with comments

Input file: data.csv
Output file: filtered_data.csv"
```

#### Code review

```
prompt:
"Please review the code below and point out improvements.
Especially in terms of performance, readability, and error handling.

```python
[Paste code]
```
"
```

#### Debugging support

```
prompt:
"The following Python code causes an error.
Please tell me the cause and how to fix it.

Error message:
KeyError: 'name'

code:
```python
[Paste the code that causes the error]
```
"
```

### 8.2.3 Data analysis and summary

#### Long summary

```
prompt:
"Please summarize the following text in about 300 characters.
Organize your main points in bullet points.

[Paste long text]
"
```

**💡 TIP**: Take advantage of MS-S1 Max's large memory capacity to summarize entire book chapters with 32K context settings.

#### Comparative analysis of data

```
prompt:
"Compare the following two product specifications and find out the advantages and disadvantages of each.
Please summarize in table format.

Product A:
[Paste specifications]

Product B:
[Paste specifications]

Comparison perspective: price, performance, scalability, support"
```

### 8.2.4 Learning support

#### Concept explanation

```
prompt:
"About "overfitting" in machine learning,
Easy to understand even for programming beginners.
Please explain using a familiar example. "
```

#### Generating practice questions

```
prompt:
"I'm currently learning the basic grammar of Python (if statements, for statements, functions).
Create three practice questions that combine these.
The difficulty level is from beginner to intermediate. Please also include model answers. "
```

## 8.3 Utilizing API server mode

### 8.3.1 Starting the local server

LM Studio can act as a local server that is OpenAI API compatible.

#### Startup procedure

```
1. Open LM Studio
2. Select “🌐 Local Server” from the tab on the left
3. Select the model you want to use
4. Click “Start Server”

Server information:
  URL: http://localhost:1234
Endpoint: /v1/chat/completions
Authentication: Not required (local environment)
```

#### Server settings

```
Settings:
Port: 1234 (default, changeable)
CORS: Enabled (accessible from browser app)
  Log Level: Info
Auto-start: Off (enable if necessary)
```

### 8.3.2 Cooperation with other applications

#### VS Code + Continue extension

**Installing Continue**

```
1. Open VS Code
2. Search for “Continue” in the extension marketplace
3. Installation
```

**Linkage settings with LM Studio**

```json
// ~/.continue/config.json

{
  "models": [
    {
      "title": "LM Studio",
      "provider": "openai",
      "model": "local-model",
      "apiBase": "http://localhost:1234/v1",
      "apiKey": "not-needed"
    }
  ],
  "tabAutocompleteModel": {
    "title": "LM Studio",
    "provider": "openai",
    "model": "local-model",
    "apiBase": "http://localhost:1234/v1"
  }
}
```

**How ​​to use**

```
1. Start Local Server in LM Studio (DeepSeek-Coder recommended)
2. Open the code in VS Code
3. Open Continue with Ctrl+L (Windows/Linux)
4. Enter questions about the code, refactoring requests, etc.
```

#### Usage from Python script

```python
import openai

# LM Studio API settings
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"

# Execute chat completion
response = openai.ChatCompletion.create(
    model="local-model",
    messages=[
{"role": "system", "content": "You are a kind assistant."},
{"role": "user", "content": "Please explain list comprehensions in Python."}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.choices[0].message.content)
```

#### Test with curl command

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

### 8.3.3 Building a web application

#### Chat app using Streamlit

```python
import streamlit as st
import openai

# LM Studio API settings
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"

st.title("Local AI Chat")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user input
if prompt := st.chat_input("Please enter your message"):
# add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Generate AI response
    with st.chat_message("assistant"):
        response = openai.ChatCompletion.create(
            model="local-model",
            messages=st.session_state.messages,
            temperature=0.7,
            stream=True
        )

        full_response = ""
        message_placeholder = st.empty()

        for chunk in response:
            if chunk.choices[0].delta.get("content"):
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

# save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
```

**How ​​to do it:**

```bash
# Install Streamlit
pip install streamlit openai

# Run the app
streamlit run chat_app.py
```

## 8.4 Efficient Workflow

### 8.4.1 Utilizing templates

Save frequently used prompts as templates.

#### Prompt template example

**Code review template**

```
Please review the code below.

[Confirmation viewpoint]
1. Bugs and potential issues
2. Room for performance improvement
3. Improved readability
4. Compliance with best practices

【code】
```
[Language name]
[Paste code]
```

[Points of particular importance]
[Example: security, performance, etc.]
```

**Text correction template**

```
Please correct the sentences below.

[Purpose] [Example: business email, blog article]
[Target audience] [Example: general users, experts]
[Style] [Example: Respectful style, regular style]

[Original text]
[Paste text]

[Correction perspective]
1. Errors in grammar and expressions
2. Suggesting more appropriate vocabulary
3. Improving sentence structure
4. Improved readability
```

### 8.4.2 Shortcuts and Tips

#### Keyboard shortcuts

```
Windows/Linux:
Ctrl+L: Focus on prompt input field
Ctrl+Enter: Send message
Ctrl+Shift+C: Copy conversation
Ctrl+Shift+N: New chat
Ctrl+Shift+H: Open hardware settings

macOS:
Cmd+L: Focus on prompt input field
Cmd+Enter: Send message
Cmd+Shift+C: Copy conversation
Cmd+Shift+N: New chat
Cmd+Shift+H: Open hardware settings
```

#### Tips for efficiency

**Tip 1: Request output in markdown format**

```
prompt:
"Please organize the following information in a Markdown format table:
[Paste data]"

→ You can paste the output directly into the document.
```

**Tip 2: Specifying code blocks**

```
prompt:
"Please provide sample code in Python.
Be sure to enclose it in a ```python`` code block. "

→ Syntax highlighted output
```

**Tip 3: Gradual output**

```
prompt:
"If the explanation is long, please separate it into sections and output it.
Please read more or confirm after each section. "

→ Easy to manage long texts
```

## 8.5 Troubleshooting

### 8.5.1 Common problems and solutions

#### Problem 1: Response stops midway

```
Cause:
- Max Tokens is too small
- Context length limit reached

Solution:
1. Increase Max Tokens (2048 → 4096)
2. Reset conversation (New Chat)
3. Set a larger context length
```

#### Problem 2: Poor response quality

```
Cause:
- Model is too small
- Unclear prompt
- Improper parameter settings

Solution:
1. Try a larger model (7B → 14B → 32B)
2. Rewrite the prompt to be more specific.
3. Adjust Temperature (too low or too high)
```

#### Problem 3: Poor Japanese quality

```
Cause:
- Uses a model that is weak in Japanese

Solution:
1. Use Qwen series (strong in Japanese)
2. Specify Japanese in system prompt
"Please be sure to respond in Japanese."
3. Provide few-shot examples
```

### 8.5.2 Performance improvements

#### Increase response speed

```
Method 1: Use a smaller model
  70B → 32B → 14B → 7B

Method 2: Lighter quantization
  Q6 → Q5 → Q4

Method 3: Reduce context length
  32K → 16K → 8K

Method 4: Change performance mode
  Quiet → Balance → Performance
```

#### Reduce memory usage

```
Method 1: Reset the conversation frequently
Method 2: Reduce context length
Method 3: Unload unnecessary models
Method 4: Close other applications
```

## 8.6 Summary of this chapter

In this chapter, you learned how to use LM Studio practically.

✅ **Chat Interface**
- Create effective prompts
- Utilize system prompts
- Manage multi-turn interactions

✅ **Practical use cases**
- Writing/editing text
- Code generation and review
- Data analysis and summarization
- Learning support

✅ **API server mode**
- Start local server
- Integration with VS Code and Python
- Web application construction

✅ **Efficient Workflow**
- Utilize templates
- shortcuts
- troubleshooting

In the next chapter, you will learn about advanced features and customization.

---

**Go to previous chapter**: [Chapter 7 Optimization settings for MS-S1 Max](chapter07_optimization.md)
**Next Chapter**: [Chapter 9 Advanced Features and Customization](chapter09_advanced_features.md)
