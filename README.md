# 🎓 AI Ignite Week - Technical Track

Complete implementation of all 7 AI/ML tasks with modern solutions using Python, Flask, and various AI APIs.

## 🚀 Project Overview

This repository contains **7 complete AI/ML projects** covering different aspects of artificial intelligence and machine learning.



### ✅ Completed Tasks

| Task | Project Name | Technology Stack | Status | Key Features |
|:----:|:-------------|:-----------------|:------:|:-------------|
| **01** | 🎯 Smart To-Do Priority Predictor | Scikit-learn, Pandas, ML | ✅ Complete | ML-based task prioritization with multiple algorithms |
| **02** | 💭 Sentiment Analyzer | TextBlob, NLP, Visualization | ✅ Complete | Customer review sentiment classification |
| **03** | ⏰ Meeting Time Tracker | Data Analysis, Matplotlib | ✅ Complete | Speaking time distribution from transcripts |
| **04** | 📧 Email Classification System | Naive Bayes, TF-IDF, NLP | ✅ Complete | Automated email categorization with 1500+ samples |
| **05** | 📚 Study Notes Summarizer | NLTK, Text Processing | ✅ Complete | Intelligent lecture note summarization |
| **06** | 🎵 Music Mood Classifier | Decision Trees, Audio ML | ✅ Complete | Song mood prediction using 114K audio features |
| **07** | 🤖 AI Study Buddy | Groq API, Flask, Web App | ✅ Complete | AI-powered question generator with web interface |



## 📂 Repository Structure### 🔍 **Task 2: Simple Sentiment Analyzer**

- **Objective**: Build a tool that analyzes customer reviews for sentiment (happy, angry, neutral)

```- **Implementation**:

AI-IGNITE-WEEK-Technical-Track/  - TextBlob-based sentiment analysis

├── Task Files/               # All Jupyter notebooks and Python files  - 50+ restaurant reviews dataset

│   ├── Task_01.ipynb        # Smart To-Do Priority Predictor  - Word frequency analysis and visualization

│   ├── Task_02.ipynb        # Sentiment Analyzer  - Sentiment distribution charts

│   ├── Task_03.ipynb        # Meeting Time Tracker- **File**: `Task_02.ipynb`

│   ├── Task_04.ipynb        # Email Classification System- **Key Features**: Real-time sentiment scoring, word cloud generation, review categorization

│   ├── Task_05.ipynb        # Study Notes Summarizer

│   ├── Task_06.ipynb        # Music Mood Classifier### ⏱️ **Task 3: Meeting Time Tracker**

│   └── Task_07.py           # AI Study Buddy (Flask Web App)- **Objective**: Analyze meeting transcripts to show who talks the most

├── All Task PDFs/           # Task documentation and requirements- **Implementation**:

├── templates/               # Web interface for Task 7  - Advanced transcript parsing algorithms

├── dataset.csv             # Music dataset for Task 6  - Speaking time calculation and visualization

├── requirements_task7.txt   # Dependencies for Task 7  - Meeting balance scoring system

├── .env.example            # Environment variables template  - Comprehensive speaker analytics

└── README.md               # This file- **File**: `Task_03.ipynb`

```- **Key Features**: Speaker time tracking, balance analysis, meeting insights



## 🛠️ Setup Instructions### 📧 **Task 4: Email Classification System**

- **Objective**: Complete email classification workflow using Naive Bayes with data visualization

### 1. Clone Repository- **Implementation**:

```bash  - 1,500+ realistic emails across 12 categories

git clone https://github.com/Muhammad-Muzammil-Shah/AI-IGNITE-WEEK-Technical-Track.git  - TF-IDF vectorization with Multinomial Naive Bayes

cd AI-IGNITE-WEEK-Technical-Track  - Comprehensive EDA with multiple visualizations

```  - Word cloud analysis and confusion matrix evaluation

- **File**: `Task_04.ipynb`

### 2. Create Virtual Environment- **Key Features**: 100% accuracy classification, model persistence, real-time predictions

```bash

python -m venv .venv### 📚 **Task 5: Study Notes Summarizer**

# Windows- **Objective**: Create short summaries of study notes for easier revision

.venv\Scripts\activate- **Implementation**:

# Linux/Mac    - Advanced NLTK-based text processing

source .venv/bin/activate  - Smart sentence scoring with keyword frequency analysis

```  - Quick 1-line and detailed 3-4 point summaries

  - File I/O operations and summary export

### 3. Install Dependencies- **File**: `Task_05.ipynb`

- **Key Features**: 6.7:1 compression ratio, readability analysis, keyword extraction

#### For Tasks 1-6 (Jupyter Notebooks):

```bash## 🛠️ Technologies Used

pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud

```- **Machine Learning**: scikit-learn, pandas, numpy

- **NLP**: NLTK, TextBlob, TF-IDF Vectorization

#### For Task 7 (Web Application):- **Visualization**: Matplotlib, Seaborn, WordCloud

```bash- **Models**: Naive Bayes, Decision Trees, Random Forest, Gradient Boosting

pip install -r requirements_task7.txt- **Data Processing**: CSV handling, text preprocessing, feature engineering

```

## 📊 Project Statistics

### 4. Setup API Keys (Task 7 only)

```bash- **Total Notebooks**: 5 comprehensive implementations

# Copy environment template- **Lines of Code**: 2,000+ lines across all tasks

cp .env.example .env- **Datasets**: 4 custom-generated realistic datasets

- **Models Trained**: 8+ different ML models

# Edit .env file and add your Groq API key- **Visualizations**: 25+ charts and graphs

# Get key from: https://console.groq.com/- **Accuracy Achieved**: Up to 100% on classification tasks

```

## 🚀 How to Run

## 🎯 Running the Projects

1. **Clone the repository**:

### Tasks 1-6: Jupyter Notebooks   ```bash

```bash   git clone https://github.com/Muhammad-Muzammil-Shah/AI-IGNITE-WEEK-Technical-Track.git

jupyter notebook   cd AI-IGNITE-WEEK-Technical-Track

# Open any Task_XX.ipynb file and run all cells   ```

```

2. **Install dependencies**:

### Task 7: Web Application   ```bash

```bash   pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud joblib

python "Task Files/Task_07.py"   ```

# Open http://localhost:5000 in your browser

```3. **Run any task**:

   ```bash

## 📋 Task Details   jupyter notebook Task_01.ipynb  # Or any other task

   ```

### 🎯 Task 1: Smart To-Do Priority Predictor

- **Technology**: Scikit-learn, Decision Trees## 📁 File Structure

- **Features**: ML-based task prioritization, accuracy comparison

- **Dataset**: 400 synthetic tasks with features```

AI-IGNITE-WEEK-Technical-Track/

### 💭 Task 2: Sentiment Analyzer  ├── Task_01.ipynb                 # Smart To-Do Priority Predictor

- **Technology**: TextBlob, NLP├── Task_02.ipynb                 # Sentiment Analyzer

- **Features**: Customer review sentiment analysis├── Task_03.ipynb                 # Meeting Time Tracker

- **Dataset**: 50 restaurant reviews├── Task_04.ipynb                 # Email Classification System

├── Task_05.ipynb                 # Study Notes Summarizer

### ⏰ Task 3: Meeting Time Tracker├── emails_dataset.csv            # Email classification dataset

- **Technology**: Data analysis, Visualization├── ai_notes.txt                  # Sample lecture notes

- **Features**: Speaking time distribution from transcripts├── climate_notes.txt             # Climate science notes

- **Output**: Charts and speaker statistics├── psychology_notes.txt          # Psychology lecture notes

├── ai_summary.txt                # Generated summary example

### 📧 Task 4: Email Classification System└── README.md                     # This file

- **Technology**: Naive Bayes, TF-IDF```

- **Features**: Email categorization with visualization

- **Dataset**: 1,500 enhanced email samples## 🎯 Key Achievements



### 📚 Task 5: Study Notes Summarizer- ✅ **All 5 tasks completed** with advanced implementations

- **Technology**: NLTK, Text summarization- ✅ **Production-ready code** with error handling and validation

- **Features**: Automatic lecture note summarization- ✅ **Comprehensive visualizations** for all analyses

- **Output**: Key points and summaries- ✅ **Real-world datasets** with realistic content

- ✅ **High performance models** with excellent accuracy scores

### 🎵 Task 6: Music Mood Classifier- ✅ **Interactive functionality** for user input processing

- **Technology**: Decision Trees, Audio analysis- ✅ **File I/O operations** for practical usage

- **Features**: Song mood prediction from audio features- ✅ **Detailed documentation** and code comments

- **Dataset**: 114,000 songs with 8 audio features

## 🔮 Future Enhancements

### 🤖 Task 7: AI Study Buddy - Question Generator

- **Technology**: Groq API, Flask, NLTK- 🚀 **Web API deployment** for real-time access

- **Features**: - 📱 **Mobile app integration** for on-the-go usage

  - AI-powered quiz generation from notes- 🤖 **Advanced deep learning models** (BERT, transformers)

  - Beautiful web interface- 🌐 **Multi-language support** for global usage

  - Keyword extraction and analysis- 📊 **Real-time dashboard** for analytics visualization

  - Interactive Q&A system

- **API**: Groq LLaMA models## 👨‍💻 Author



## 🌟 Key Features**Muhammad Muzammil Shah**

- AI/ML Engineer

### 🎨 Modern Web Interface (Task 7)- Specializing in NLP and Machine Learning Applications

- Responsive design with gradient backgrounds

- Interactive quiz system## 📄 License

- Real-time keyword analysis

- Smart question categorizationThis project is open source and available under the [MIT License](LICENSE).



### 🤖 AI Integration---

- **Groq API**: Advanced language models for question generation

- **NLTK**: Natural language processing**🎓 AI Ignite Week Technical Track - Complete Implementation**

- **Scikit-learn**: Machine learning algorithms*Transforming ideas into intelligent solutions with cutting-edge machine learning!*

### 📊 Data Visualization
- Beautiful charts and graphs
- Interactive visualizations
- Statistical analysis
- Performance metrics

### 🔒 Security Features
- Environment variable configuration
- API key protection
- Error handling and fallbacks

## 🚨 Important Notes

### API Requirements
- **Task 7** requires a Groq API key (free tier available)
- All other tasks work offline

### Datasets
- Most datasets are synthetic/generated for educational purposes
- Task 6 uses a real music dataset with 114K songs
- All datasets included in repository

### Performance
- Task 6 achieves realistic 75-85% accuracy (fixed from 100% overfitting)
- All models optimized for educational demonstration
- Production-ready error handling

## 🔧 Troubleshooting

### Common Issues
1. **NLTK Data Missing**: Run any notebook once to auto-download
2. **Groq API Errors**: Check your API key in `.env` file
3. **Flask Import Errors**: Install requirements: `pip install -r requirements_task7.txt`
4. **Jupyter Issues**: Try: `pip install jupyter notebook`

### Getting Help
- Check individual task files for specific instructions
- Review error messages for debugging hints
- All tasks have fallback options for offline use

## 🎓 Educational Value

This project demonstrates:
- **Machine Learning**: Classification, regression, feature engineering
- **Natural Language Processing**: Text analysis, sentiment analysis, summarization
- **Web Development**: Flask applications, responsive design
- **Data Science**: Visualization, statistical analysis
- **AI Integration**: Modern AI APIs, prompt engineering

## 🚀 Future Enhancements

- [ ] Docker containerization
- [ ] REST API for all tasks
- [ ] Database integration
- [ ] User authentication
- [ ] Mobile app versions
- [ ] Advanced AI models

## 📞 Contact

**Muhammad Muzammil Shah**
- GitHub: [@Muhammad-Muzammil-Shah](https://github.com/Muhammad-Muzammil-Shah)
- Repository: [AI-IGNITE-WEEK-Technical-Track](https://github.com/Muhammad-Muzammil-Shah/AI-IGNITE-WEEK-Technical-Track)

---

**🎯 AI Ignite Week - Technical Track**  
*Complete AI/ML project showcase with modern technologies* 🚀