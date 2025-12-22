# Chapter 4: Basic operations and interface

## 4.1 Main screen configuration

```
Screen layout:
├── Left sidebar: Mode selection
├── Center: Main work area
├── Right sidebar: Parameters
└── Top tabs: Model, Parameters, Extensions, etc.
```

## 4.2 Chat mode

### Basic usage

```
1. Select "Chat" tab
2. Enter your message in the text box
3. "Generate" or Enter key
4. Wait for response
```

### Useful features

```
✓ Continue: Continue generation
✓ Regenerate: Regenerate
✓ Remove last: Delete the last message
✓ Copy: Copy to clipboard
✓ Replace last reply: Edit response
```

## 4.3 Notebook mode

### Long writing

```
Usage:
- Novels/Scenarios
- Technical documentation
- Blog article

operation:
1. "Notebook" tab
2. Initial text in text area
3. Generate the rest with "Generate"
4. Click "Continue" to continue further.
```

## 4.4 Parameter basics

### Main parameters

```python
# Temperature: Creativity
0.1-0.5 # Solid, fact-oriented
0.7-0.9 # Balance
1.0-1.5 # creative

# Top-P: Diversity
0.9 # standard
0.95 # Moderately diverse

# Top-K: Limit number of candidates
40# standard
100 # variety
```

## 4.5 Saving presets

```
1. Adjust parameters
2. "Parameters" tab
3. "Save" button
4. Enter preset name
5. Call later from "Load"
```

## 4.6 Managing conversations

### Save and load

```
keep:
"Chat" tab → "Save history" → Enter file name

Load:
"Chat" tab → "Load history" → File selection
```

## 4.7 Summary of this chapter

✅ Main screen configuration
✅ Chat/Notebook mode
✅ Basic parameters
✅ Presets and conversation management

---

**Go to previous chapter**: [Chapter 3 ROCm settings and ExLlamaV2 optimization](chapter03_rocm_exllama.md)
**Next Chapter**: [Chapter 5 Model Loaders and Format](chapter05_model_loaders.md)
