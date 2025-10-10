"""
AI Ignite Week - Task 7: AI Study Buddy - Question Generator
A Flask web application that generates quiz questions from lecture notes using Groq API
"""

from flask import Flask, render_template, request, jsonify
import nltk
import re
import random
import os
from collections import Counter
import string

app = Flask(__name__)

# Initialize Groq client with error handling
try:
    from groq import Groq
    # Get API key from environment variable
    api_key = os.getenv('GROQ_API_KEY', 'your_groq_api_key_here')
    if api_key == 'your_groq_api_key_here':
        print("⚠️ Please set GROQ_API_KEY environment variable")
        print("💡 Or replace 'your_groq_api_key_here' with your actual API key")
        client = None
        GROQ_AVAILABLE = False
    else:
        client = Groq(api_key=api_key)
        GROQ_AVAILABLE = True
        print("✅ Groq API initialized successfully")
except Exception as e:
    print(f"⚠️ Groq API initialization failed: {e}")
    print("🔄 Will use fallback question generation")
    client = None
    GROQ_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

class StudyBuddy:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_keywords(self, text, top_n=10):
        """Extract important keywords from text"""
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = word_tokenize(text)
        
        # Remove stopwords and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Get word frequency
        word_freq = Counter(words)
        
        # Get POS tags to focus on nouns, adjectives, and important verbs
        pos_tags = pos_tag(words)
        
        # More specific POS filtering for better keywords
        important_words = []
        for word, pos in pos_tags:
            # Focus on nouns, proper nouns, adjectives, and key verbs
            if pos.startswith(('NN', 'JJ')) or pos in ['VB', 'VBG', 'VBN']:
                # Skip very common words even if they're not in stopwords
                if word not in ['use', 'used', 'using', 'make', 'made', 'get', 'got', 'way', 'ways', 'thing', 'things', 'time', 'times']:
                    important_words.append(word)
        
        # Combine frequency and POS importance with better scoring
        keyword_scores = {}
        for word in important_words:
            if word in word_freq and word_freq[word] > 1:  # Must appear at least twice
                # Score based on frequency and word length (longer words often more important)
                score = word_freq[word] * (len(word) / 10 + 1)
                keyword_scores[word] = score
        
        # Sort by score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter out very similar words (basic deduplication)
        final_keywords = []
        for word, score in sorted_keywords:
            # Check if similar word already exists
            is_similar = False
            for existing in final_keywords:
                if (word in existing or existing in word) and abs(len(word) - len(existing)) < 3:
                    is_similar = True
                    break
            
            if not is_similar:
                final_keywords.append(word)
                
            if len(final_keywords) >= top_n:
                break
        
        return final_keywords
    
    def extract_important_sentences(self, text, top_n=5):
        """Extract important sentences from text"""
        sentences = sent_tokenize(text)
        
        # Score sentences based on keyword density
        keywords = self.extract_keywords(text, 20)
        keyword_set = set(keywords)
        
        sentence_scores = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = sum(1 for word in words if word in keyword_set)
            if len(words) > 5:  # Ignore very short sentences
                sentence_scores.append((sentence, score))
        
        # Sort by score and return top sentences
        sentence_scores = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
        return [sentence for sentence, score in sentence_scores[:top_n]]
    
    def generate_questions_with_groq(self, text, num_questions=5):
        """Generate MCQ questions using Groq API"""
        # Extract key information
        keywords = self.extract_keywords(text, 15)
        important_sentences = self.extract_important_sentences(text, 8)
        
        # Check if Groq is available
        if not GROQ_AVAILABLE or client is None:
            print("Using fallback question generation (Groq not available)")
            return self.generate_simple_questions(keywords, important_sentences, num_questions)
        
        # Create prompt for Groq
        prompt = f"""
Based on the following lecture notes, generate {num_questions} multiple-choice questions for studying.

LECTURE NOTES:
{text[:2000]}  # Limit text length for API

KEY TOPICS: {', '.join(keywords)}

REQUIREMENTS:
1. Create exactly {num_questions} multiple-choice questions
2. Each question should have 4 options (a, b, c, d)
3. Include the correct answer
4. Focus on important concepts and definitions
5. Make questions educational and clear
6. Cover different topics from the notes

FORMAT EXAMPLE:
Q1: What is Machine Learning?
(a) A type of computer hardware
(b) A subset of Artificial Intelligence
(c) A programming language
(d) A database system
Answer: (b)

Generate the questions now:
"""
        
        try:
            # Call Groq API
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an educational AI assistant that creates high-quality study questions from lecture notes."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",  # or "mixtral-8x7b-32768"
                temperature=0.7,
                max_tokens=1500
            )
            
            return self.parse_groq_response(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            # Fallback to simple questions if API fails
            return self.generate_simple_questions(keywords, important_sentences, num_questions)
    
    def parse_groq_response(self, response_text):
        """Parse Groq API response into structured questions"""
        questions = []
        lines = response_text.strip().split('\n')
        
        current_question = {}
        question_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for question start
            if line.startswith(('Q', 'q')) and ':' in line:
                if current_question:
                    questions.append(current_question)
                    current_question = {}
                
                question_count += 1
                current_question['id'] = question_count
                current_question['question'] = line.split(':', 1)[1].strip()
                current_question['options'] = []
                
            # Check for options
            elif line.startswith(('(a)', '(b)', '(c)', '(d)', 'a)', 'b)', 'c)', 'd)')):
                if current_question:
                    option = line[3:].strip() if line.startswith('(') else line[2:].strip()
                    current_question['options'].append(option)
                    
            # Check for answer
            elif line.lower().startswith('answer'):
                if current_question:
                    answer_part = line.split(':', 1)[1].strip() if ':' in line else line.split()[1]
                    current_question['answer'] = answer_part.replace('(', '').replace(')', '').strip()
        
        # Add last question if exists
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def generate_simple_questions(self, keywords, sentences, num_questions=5):
        """Fallback method to generate simple questions"""
        questions = []
        used_keywords = set()
        
        # Question templates
        templates = [
            "{keyword} is related to which field?",
            "What is {keyword}?",
            "Which of the following describes {keyword}?",
            "{keyword} is primarily used for?",
            "The main purpose of {keyword} is?"
        ]
        
        # Generate options pool
        all_keywords = keywords + ['Data Science', 'Technology', 'Research', 'Innovation', 'Analysis']
        
        for i in range(min(num_questions, len(keywords))):
            keyword = keywords[i]
            if keyword in used_keywords:
                continue
                
            used_keywords.add(keyword)
            
            question_text = random.choice(templates).format(keyword=keyword.title())
            
            # Create options
            correct_option = keyword.title()
            wrong_options = [kw.title() for kw in all_keywords if kw != keyword]
            random.shuffle(wrong_options)
            wrong_options = wrong_options[:3]
            
            all_options = [correct_option] + wrong_options
            random.shuffle(all_options)
            
            correct_answer = chr(97 + all_options.index(correct_option))  # a, b, c, d
            
            questions.append({
                'id': i + 1,
                'question': question_text,
                'options': all_options,
                'answer': correct_answer
            })
        
        return questions

# Initialize StudyBuddy
study_buddy = StudyBuddy()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        notes_text = data.get('notes', '').strip()
        num_questions = int(data.get('num_questions', 5))
        
        if not notes_text:
            return jsonify({'error': 'Please provide lecture notes'}), 400
        
        if len(notes_text) < 50:
            return jsonify({'error': 'Notes too short. Please provide more detailed content.'}), 400
        
        # Generate questions
        questions = study_buddy.generate_questions_with_groq(notes_text, num_questions)
        
        if not questions:
            return jsonify({'error': 'Failed to generate questions. Please try again.'}), 500
        
        # Extract keywords for display
        keywords = study_buddy.extract_keywords(notes_text, 10)
        
        return jsonify({
            'success': True,
            'questions': questions,
            'keywords': keywords,
            'total_questions': len(questions)
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/analyze_notes', methods=['POST'])
def analyze_notes():
    try:
        data = request.get_json()
        notes_text = data.get('notes', '').strip()
        
        if not notes_text:
            return jsonify({'error': 'Please provide lecture notes'}), 400
        
        # Extract keywords and important sentences
        keywords = study_buddy.extract_keywords(notes_text, 15)
        important_sentences = study_buddy.extract_important_sentences(notes_text, 5)
        
        # Basic statistics
        word_count = len(notes_text.split())
        sentence_count = len(sent_tokenize(notes_text))
        
        return jsonify({
            'success': True,
            'keywords': keywords,
            'important_sentences': important_sentences,
            'statistics': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'keyword_count': len(keywords)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("🎓 AI Study Buddy - Question Generator")
    print("=" * 50)
    
    if not GROQ_AVAILABLE:
        print("⚠️  GROQ API NOT AVAILABLE")
        print("📦 To fix this, run: pip install groq==0.11.0")
        print("🔑 Or set GROQ_API_KEY environment variable")
        print("� Using fallback question generation for now")
        print("-" * 50)
    
    print("�📚 Ready to help you study!")
    print("🌐 Open http://localhost:5000 in your browser")
    print("💡 Tip: Use detailed lecture notes for better questions")
    print("=" * 50)
    
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("🔧 Try: pip install flask==2.3.3")
