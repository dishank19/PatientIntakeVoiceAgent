 **Neurality Health AI — Voice Agent MVP (Take-Home Project)**

**🩺 Overview**  
 At Neurality Health AI, we're revolutionizing patient engagement by building multimodal AI agents, starting with voice-first interfaces. This take-home project invites you to develop a simplified prototype of a **voice-based AI assistant** capable of understanding and responding to patient requests in multiple languages.

**🎯 Objective**  
 Develop a basic Voice AI Agent that can:

1. **Transcribe** a patient's audio message (supporting multiple languages).

2. **Classify** the intent of the message (e.g., appointment scheduling, billing inquiry, prescription refill).

3. **Generate** an appropriate response using a Generative AI model.

4. **Output** the result as a structured JSON object.

**🧪 Sample Input**  
 An audio file in `.mp3` or `.wav`, for example:

Spanish: “Hola, quiero saber si mi seguro cubre limpiezas dentales.”

**✅ Expected Output**  
 A JSON response like:

json  
CopyEdit  
`{`  
  `"transcript": "Hola, quiero saber si mi seguro cubre limpiezas dentales.",`  
  `"language": "Spanish",`  
  `"intent": "insurance_coverage_inquiry",`  
  `"response": "Claro, puedo ayudarte. ¿Podrías indicarme el nombre de tu proveedor de seguros?",`  
  `"confidence_score": 0.91`  
`}`

**🧰 Tools & Technologies (Suggestions)**

* **Speech-to-Text**: OpenAI Whisper, Vosk, Google Speech-to-Text, Deepgram, etc.  
  * You can also use a S2S model

* **Intent Classification**: Rule-based methods, scikit-learn, or fine-tuned models.

* **Response Generation**: GPT-4 or open-source LLMs via [Pipecat V3](https://github.com/pipecat-ai/pipecat).

* **Programming Languages**: Python preferred.

* **Frameworks**: FastAPI for API development (optional).

**🛠 Deliverables**

* A script or notebook that processes a sample audio input end-to-end.

* An output file or API endpoint that returns a JSON response.

* A short README detailing:

  * Tools and libraries used.

  * Assumptions or shortcuts taken.

  * Considerations for scaling this into a production environment within healthcare settings.

**🕐 Time Estimate**  
 Approximately 4–6 focused hours.  
 Emphasis is on **thought process and approach** rather than complete polish.

**🌟 Bonus (Optional)**

* ***EXTRA BONUS: Integrate with a phone number (use twilio or Retell)***  
* Implement multilingual support with automatic language detection.

* Deploy as a simple FastAPI endpoint.

* Create a basic UI to upload audio and display results.

**🧠 Why This Matters**  
 Voice-based patient interaction is the *first mile* of modern care delivery. This challenge reflects the core of what we’re building at Neurality — an AI-powered operating system that improves patient access and reduces staff burden.

