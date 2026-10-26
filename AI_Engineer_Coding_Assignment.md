# Coding Assignment – AI Engineer

## Objective

Design and implement a **Python-based AI chatbot** that can answer user questions about **companies** using a **static dataset**, while maintaining **conversation context**, handling **ambiguous queries through follow-up questions**, and demonstrating **product-quality engineering practices** such as sanitization, logging, and observability.

The purpose of this assignment is to evaluate:
- Problem understanding
- AI / LLM system design
- Data handling and reasoning
- Code quality
- Product and engineering mindset

---

## Dataset

Use the following **static dataset** as the knowledge source:

**Kaggle Dataset:**  
https://www.kaggle.com/datasets/srisankargiri/list-of-118-unicorn-startups-in-indiamay-2025

The dataset contains information about Indian unicorn startups such as company name, sector, and location.

> **Note:**  
> - No live web search is required.  
> - The chatbot must rely only on this dataset (and any preprocessing you perform).

---

## Functional Requirements

### 1. Chatbot Capabilities

The chatbot should allow users to:
- Ask factual questions about companies  
  - Example: *“What does Razorpay do?”*
- Ask exploratory or filtering questions  
  - Example: *“List fintech unicorn startups.”*
- Ask subjective or vague questions  - should use knowledge matching
  - Example: *“Which company would be a good fit for collaboration?”*

---

### 2. Conversation History

- The chatbot must maintain **conversation context** across multiple turns.
- Follow-up questions should consider prior user inputs and responses.
- Please use GenAI stack.

**Example:**
User: Tell me about fintech unicorns
User: Which of these are based in Bangalore?


---

### 3. Cross-Questioning / Clarification

- If a user query is **ambiguous or underspecified**, the chatbot should ask a **clarifying question** instead of guessing.

**Example:**
User: Which company is best to collaborate with?
Bot: Can you clarify the type of collaboration you are looking for (e.g., technology, payments, logistics)?


---

## Non-Functional / Engineering Requirements

### 4. Query Sanitization

Implement basic sanitization, such as:
- Input trimming and normalization
- Handling malformed or irrelevant inputs
- Basic protection against prompt injection or harmful patterns

---

### 5. Logging

The system should log:
- User queries
- Bot responses
- Timestamps
- Session or conversation identifiers

Logs should be structured and suitable for debugging or analysis.

---

### 6. Observability / Metrics (Lightweight)

Demonstrate **product thinking** by capturing at least some of the following:
- Number of queries handled
- Number of clarification questions triggered
- Errors or fallback responses
- Response latency (optional)
- Please use GenAI stack (e.g., Langfuse)
  
> Full production monitoring is not required, but design intent should be clear.

---

## Technical Constraints

- **Programming Language:** Python
- **Data Source:** Static dataset only (provided above)
- **Frameworks/Libraries:** Candidate’s choice  
  (e.g., FastAPI, Flask, LangChain, sentence-transformers, OpenAI API, etc.)

---

## Deliverables

1. **Source Code**
   - Well-structured and readable
   - Clear separation of concerns (data loading, chatbot logic, logging, etc.)

2. **README.md**
   Should include:
   - Setup instructions
   - How to run the chatbot
   - High-level architecture and design decisions
   - Assumptions and limitations

3. **Sample Interaction**
   - Provide a few example user queries and chatbot responses

---

## Evaluation Criteria

Candidates will be evaluated on:

| Area | What We Look For |
|------|------------------|
| Problem Understanding | Correct interpretation of requirements |
| AI Design | Thoughtful use of LLMs / retrieval / reasoning |
| Conversation Handling | Context awareness & clarification logic |
| Code Quality | Clean, modular, readable Python code |
| Product Thinking | Logging, sanitization, observability |
| Practicality | Sensible assumptions and trade-offs |

---

## Notes to Candidates

- This is **not a UI/UX exercise** — a CLI or simple API is sufficient.
- We value **design clarity and reasoning** over feature completeness.
- You may document any shortcuts taken and improvements you would make with more time.

---
