# ALI STUDY SAMPLE VIDEOS — HACKATHON DATASET

**Last Updated:** November 2025
**Version:** 1.0
**Dataset Type:** Educational Video Samples (Demonstration)

---

## 📋 PURPOSE OF THIS DATASET

This folder contains **two instructional video samples** from the **ALI Vocabulary Intervention Study**, a remote randomized controlled trial evaluating whether combining text-supplemented audiobooks with one-on-one remote scaffolding improves vocabulary learning outcomes among 3rd–4th grade students.

### Intended Use

These files are provided **exclusively for experimental use within this hackathon**. They are intended to support research and prototyping in:

- 🤖 Multimodal educational AI
- 📊 Learning analytics
- 🔒 Privacy-preserving machine learning

> **Important:** This dataset is a **demonstration sample**, not a comprehensive or benchmark corpus.

---

## 📚 SOURCE STUDY (REQUIRED CITATION)

**If you use this dataset, you must cite:**

```bibtex
@article{olson2025ali,
  title={Remote text-supplemented audiobook intervention supports children's
         explicit and incidental vocabulary learning},
  author={Olson, H. A. and Ozernov-Palchik, O. and Arechiga, X. M. and
          Gabrieli, J. D. E.},
  journal={Preprint},
  year={2025},
  note={Updated Oct 22, 2025. Source: ALI Study Dataset},
  url={https://osf.io/preprints/psyarxiv/[ID]}
}
```

**APA Format:**
> Olson, H.A., Ozernov-Palchik, O., Arechiga, X.M., & Gabrieli, J.D.E. (2025). Remote text-supplemented audiobook intervention supports children's explicit and incidental vocabulary learning. *Preprint*, updated Oct 22, 2025. [Source: ALI Study Dataset]

---

## ⚖️ ETHICAL USE AND COMPLIANCE REQUIREMENTS

> **⚠️ CRITICAL:** These videos include interactions with **minors** and were collected under **IRB-approved research protocols**. Use is subject to strict ethical guidelines.

### ✅ ALLOWED Uses

- Local algorithm development **within the hackathon**
- Research on:
  - Speaker diarization
  - Emotion and sentiment signals
  - Anonymization techniques
  - Instructional analytics
  - Learning signal modeling
- Demonstrations **within this event only**

### ❌ NOT ALLOWED

- ⛔ Training or fine-tuning **generalizable** machine learning models
- ⛔ Uploading to public AI systems, cloud transcription services, or external repositories
- ⛔ Republishing or redistributing any media, metadata, transcripts, embeddings, or derivatives
- ⛔ Attempting to identify or profile participants
- ⛔ Commercial use of any kind
- ⛔ Long-term storage beyond hackathon duration

### 🔐 Privacy Standard

**When in doubt, assume the strictest privacy standard.**

All work must comply with:
- FERPA (Family Educational Rights and Privacy Act)
- COPPA (Children's Online Privacy Protection Act)
- IRB ethical research standards

---

## 🎯 HACKATHON TRACKS AND RECOMMENDED USES

Teams may explore **one or more** of the following research tracks. Each represents an active research frontier relevant to educational AI.

### Track A: Speaker Diarization

**Challenge:** Detect who is speaking, segment speech turns, and align to transcript timestamps.

**Research Questions:**
- Can you accurately distinguish between tutor and student speech?
- How do you handle overlapping speech or interruptions?
- Can you identify turn-taking patterns that indicate engagement?

**Potential Outputs:**
- Timestamped speaker labels
- Turn-taking visualizations
- Talk time balance metrics

---

### Track B: Privacy-Preserving Anonymization

**Challenge:** Blur faces, mask voices, and automatically detect identifying content such as names or locations.

**Research Questions:**
- Can you detect and redact personally identifiable information (PII)?
- How do you balance privacy protection with analytical utility?
- Can you anonymize while preserving educational signals?

**Potential Outputs:**
- Automated face blurring pipelines
- Voice anonymization models
- PII detection and masking systems

---

### Track C: Sentiment and Emotion Analysis

**Challenge:** Identify learner affect such as frustration, confusion, interest, motivation, or confidence.

**Research Questions:**
- Can you detect emotional states from audio, video, or multimodal signals?
- How does learner affect change during instruction?
- Can you identify moments of breakthrough or struggle?

**Potential Outputs:**
- Emotion timeline visualizations
- Affect classification models
- Engagement prediction systems

---

### Track D: Learning-State Prediction

**Challenge:** Analyze whether the learner is gaining understanding, stuck, or requires scaffolding.

**Research Questions:**
- Can you predict when a student needs help?
- What signals indicate comprehension vs. confusion?
- How do you distinguish productive struggle from frustration?

**Potential Outputs:**
- Real-time learning state classifiers
- Knowledge acquisition models
- Scaffolding recommendation systems

---

### Track E: Instructional Quality Analytics

**Challenge:** Measure instructional patterns including talk balance, wait-time duration, scaffolding techniques, and encouragement frequency to generate feedback summaries.

**Research Questions:**
- What makes effective tutoring in this context?
- Can you quantify instructional quality metrics?
- How do different tutoring strategies affect learning?

**Potential Outputs:**
- Instructional quality dashboards
- Tutor feedback reports
- Pedagogical pattern detection

---

## 📁 FILE CONTENTS

```
ALI_Hackathon_Dataset/
├── ALI_video_sample_01.mp4    (~10-15 min, standard scaffolded tutoring)
├── ALI_video_sample_02.mp4    (~10-15 min, alternate facilitator style)
└── README.md                   (this file)
```

### Video Descriptions

**Sample 01:** Standard scaffolded tutoring example
- Demonstrates typical vocabulary instruction session
- Shows common scaffolding techniques
- Includes multiple learning opportunities

**Sample 02:** Alternate facilitator style and learner profile
- Different tutoring approach
- Varied learner characteristics
- Complementary instructional patterns

> **Note:** No transcripts are included to enable exploration of automated transcription, diarization, and speech processing pipelines.

---

## 🛠️ TECHNICAL STARTING POINTS (OPTIONAL GUIDANCE)

### Recommended Frameworks and Tools

#### Speech and Diarization
- **Whisper** (OpenAI) - State-of-the-art speech recognition
- **PyAnnote** - Speaker diarization toolkit
- **SpeechBrain** - All-in-one speech processing
- **NeMo** (NVIDIA) - Conversational AI toolkit

#### Multimodal Analysis
- **Audio embeddings** - Wav2Vec, HuBERT
- **Pose estimation** - MediaPipe, OpenPose
- **NLP sentiment** - BERT, RoBERTa fine-tuned for education
- **Video understanding** - TimeSformer, VideoMAE

#### Privacy and Anonymization
- **Face detection/blurring** - OpenCV, MediaPipe Face Detection
- **Voice conversion** - SoX, Praat
- **PII detection** - spaCy NER, custom education-specific models

#### Learning Analytics
- **Conversation analysis** - Turn-taking metrics, discourse parsing
- **Time-series models** - LSTM, Transformer for learning states
- **Rule-based frameworks** - Education research-informed heuristics

### Quick Start Code Snippets

**Python Environment Setup:**
```bash
pip install openai-whisper pyannote.audio speechbrain opencv-python mediapipe
```

**Basic Transcription (Whisper):**
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("ALI_video_sample_01.mp4")
print(result["text"])
```

**Speaker Diarization (PyAnnote):**
```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization = pipeline("ALI_video_sample_01.mp4")
```

**Face Blurring (OpenCV):**
```python
import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Process video frames and blur detected faces
```

---

## 📊 EVALUATION CRITERIA (SUGGESTED)

Hackathon projects will be evaluated on:

1. **Innovation** (25%)
   - Novel approaches to educational AI challenges
   - Creative use of multimodal signals

2. **Technical Execution** (25%)
   - Code quality and reproducibility
   - Model performance and accuracy

3. **Educational Impact** (25%)
   - Potential to improve learning outcomes
   - Alignment with educational research

4. **Ethics and Privacy** (25%)
   - Responsible data handling
   - Privacy-preserving techniques
   - Compliance with guidelines

---

## 📞 CONTACT

### For Hackathon Support
- **Email:** oozernov@bu.edu
- **Institution:** Boston University / MIT McGovern Institute

### For Research Inquiries
For inquiries beyond hackathon use, redistribution requests, or research collaborations, please contact the principal investigators.

### Data Access After Hackathon
These videos are provided for **hackathon use only**. For ongoing research access to ALI study data:
1. Submit formal data use agreement request
2. Obtain institutional IRB approval
3. Provide detailed research proposal
4. Agree to data protection protocols

---

## 🙏 ACKNOWLEDGMENTS

This dataset was developed through the **ALI research initiative** with support from:

- **Chan Zuckerberg Initiative** - Primary funding
- **National Science Foundation (NSF)** - Educational research support
- **National Institutes of Health (NIH)** - Learning sciences research
- **Learning Ally** - Digital reading platform access and partnership

### Special Thanks

We acknowledge **participating families** whose contribution enables responsible innovation in educational technology. Their trust and participation make this research possible.

We also thank:
- Research assistants and tutors who conducted the interventions
- School district partners who facilitated recruitment
- Technology partners who enabled remote delivery
- The broader research community advancing ethical AI in education

---

## 📄 LICENSE AND TERMS OF USE

### License Type
**Research Use Only - Restricted License**

### Terms
1. **Scope:** This data may be used only for hackathon activities as described above
2. **Duration:** Access rights terminate at hackathon conclusion unless extended by written agreement
3. **Attribution:** All uses must cite the source study (see citation above)
4. **Derivatives:** No derivative datasets may be created or shared without explicit permission
5. **Deletion:** All copies must be deleted within 30 days of hackathon conclusion
6. **No Warranty:** Data is provided "as is" without warranty of any kind

### Legal Disclaimer

THIS DATASET IS PROVIDED FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY. THE AUTHORS, INSTITUTIONS, AND FUNDING AGENCIES MAKE NO WARRANTIES ABOUT THE COMPLETENESS, RELIABILITY, OR ACCURACY OF THIS DATA. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY ARISING FROM THE USE OF THIS DATASET.

---

## 📋 APPENDIX: STUDY BACKGROUND

### The ALI Vocabulary Intervention

**Study Design:** Remote randomized controlled trial (RCT)

**Participants:** 3rd–4th grade students (ages 8-10)

**Intervention Components:**
1. **Text-supplemented audiobooks** - Synchronized audio and text
2. **One-on-one remote scaffolding** - Live tutoring sessions
3. **Explicit vocabulary instruction** - Direct teaching of target words
4. **Incidental learning opportunities** - Exposure to rich language context

**Key Findings:**
- Significant improvements in both explicit and incidental vocabulary learning
- Remote delivery maintained intervention fidelity
- Text supplementation enhanced comprehension and engagement
- Individual scaffolding critical for struggling learners

**Research Questions Addressed:**
- Can remote interventions match in-person effectiveness?
- How does text supplementation support audiobook learning?
- What scaffolding techniques most effectively support vocabulary acquisition?

### Implications for Educational AI

This dataset represents real-world educational interactions, offering opportunities to:
- Develop AI tools that support rather than replace human tutors
- Create privacy-preserving learning analytics
- Build adaptive scaffolding systems
- Understand multimodal signals of learning

---

## 🚀 GETTING STARTED CHECKLIST

Before beginning your hackathon project:

- [ ] Read and understand ethical use requirements
- [ ] Set up secure local development environment
- [ ] Install recommended technical frameworks
- [ ] Review sample videos to understand content
- [ ] Select one or more research tracks
- [ ] Plan privacy-preserving analysis approach
- [ ] Prepare to cite source study in all outputs
- [ ] Schedule data deletion for post-hackathon

---

## 📚 RECOMMENDED READING

### Educational AI and Learning Sciences
- Pane, J. F., et al. (2023). Adaptive educational technologies: A review
- D'Mello, S., & Graesser, A. (2012). Dynamics of affective states during learning
- Chi, M. T., & Wylie, R. (2014). The ICAP framework

### Privacy in Educational Data
- Daries, J. P., et al. (2014). Privacy, anonymity, and big data in the social sciences
- Prinsloo, P., & Slade, S. (2017). Ethics and learning analytics

### Multimodal Learning Analytics
- Blikstein, P., & Worsley, M. (2016). Multimodal learning analytics
- Ochoa, X., & Worsley, M. (2016). Augmenting learning analytics with multimodal sensory data

---

## ❓ FREQUENTLY ASKED QUESTIONS

**Q: Can I upload videos to cloud services for transcription?**
A: No. All processing must be done locally to maintain privacy protections.

**Q: Can I share my trained models publicly?**
A: You may share model architectures and code, but not trained weights if they encode information from the videos.

**Q: What if I accidentally identify a participant?**
A: Immediately delete any identifying information and notify the organizers. Do not use or share such information.

**Q: Can I use these videos for a class project after the hackathon?**
A: No. Access is limited to hackathon duration only. Contact the researchers for extended access.

**Q: Are there additional ALI study videos available?**
A: Additional data may be available through formal data sharing agreements with appropriate IRB approval.

---

## 📅 VERSION HISTORY

- **v1.0** (November 2025) - Initial hackathon release
  - 2 sample videos
  - Comprehensive documentation
  - Ethical use guidelines

---

## 🏁 CONCLUSION

Thank you for participating in responsible innovation in educational technology. Your work has the potential to improve learning outcomes for thousands of students while maintaining the highest standards of privacy and ethics.

**Remember:**
- Prioritize privacy and ethics in all work
- Cite the source study in all outputs
- Delete all data after hackathon
- Reach out with questions

**Good luck, and happy hacking!**

---

*This README was prepared by the ALI Research Team*
*Boston University & MIT McGovern Institute for Brain Research*
*November 2025*

---

**END OF DOCUMENT**
