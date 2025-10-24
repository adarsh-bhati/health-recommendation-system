from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_sqlalchemy import SQLAlchemy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "superphenomena")

# ✅ UPDATED: PostgreSQL Configuration for Render
database_url = os.environ.get('DATABASE_URL', 'sqlite:///healthdata.db')

# Fix PostgreSQL URL format for Render
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------ DATABASE MODELS ------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(10), unique=True, nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # ✅ NEW: Admin flag

class HealthInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    weight = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    activity_level = db.Column(db.String(50), nullable=False)
    goal = db.Column(db.String(50), nullable=False)

# ------------------ ADMIN AUTHENTICATION ------------------
def admin_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login to continue!")
            return redirect(url_for('login'))
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash("🔒 Admin access required!")
            return redirect(url_for('dashboard'))
        
        return func(*args, **kwargs)
    return wrapper

# ------------------ ENHANCED HEALTH CHATBOT ------------------
class HealthChatbot:
    def __init__(self):
        self.health_knowledge = {
            # General Health
            "general_health": [
                "Maintain a balanced diet with fruits, vegetables, and whole grains.",
                "Drink at least 8 glasses of water daily for proper hydration.",
                "Get 7-9 hours of quality sleep every night for optimal health.",
                "Practice good hygiene and wash hands regularly to prevent infections.",
                "Avoid smoking and limit alcohol consumption for better health.",
                "Manage stress through meditation, yoga, or deep breathing exercises.",
                "Get regular health check-ups to monitor your overall health status."
            ],
            
            # Diet & Nutrition
            "diet_nutrition": [
                "Eat 5-6 small meals throughout the day instead of 3 large ones for better metabolism.",
                "Include lean protein in every meal - chicken, fish, eggs, lentils, tofu.",
                "Consume healthy fats like avocado, nuts, seeds, and olive oil in moderation.",
                "Limit processed foods, sugary drinks, and high-sodium foods.",
                "Eat plenty of fiber-rich foods like whole grains, fruits, and vegetables for better digestion.",
                "Include calcium-rich foods like milk, yogurt, and leafy greens for bone health.",
                "Eat a rainbow of fruits and vegetables to get various vitamins and minerals."
            ],
            
            # Exercise & Fitness
            "exercise_fitness": [
                "Aim for 150 minutes of moderate exercise or 75 minutes of vigorous exercise per week.",
                "Include both cardio (running, cycling) and strength training (weight lifting) in your routine.",
                "Warm up for 5-10 minutes before exercise and cool down afterwards to prevent injuries.",
                "Stay active throughout the day - take walking breaks if you have a sedentary job.",
                "Include flexibility exercises like yoga or stretching to improve mobility.",
                "Listen to your body and don't overexert yourself - rest when needed.",
                "Find physical activities you enjoy to make exercise a sustainable habit."
            ],
            
            # Mental Health
            "mental_health": [
                "Practice mindfulness and meditation for at least 10-15 minutes daily.",
                "Take regular breaks and manage stress through hobbies and relaxation techniques.",
                "Maintain social connections with friends and family for emotional support.",
                "Seek professional help if feeling overwhelmed, anxious, or depressed.",
                "Practice gratitude by writing down things you're thankful for each day.",
                "Get sunlight exposure daily to boost vitamin D and improve mood.",
                "Limit screen time and take digital detox breaks for mental clarity."
            ],
            
            # Weight Management
            "weight_management": [
                "BMI is body mass index which defines your body mass if you want your bmi fill the healthform and check your bmi",
                "BMI between 18.5-24.9 is considered healthy range for most adults.",
                "Lose weight gradually - 1-2 pounds per week is safe and sustainable.",
                "Combine balanced diet with regular exercise for best weight loss results.",
                "Focus on lifestyle changes rather than quick-fix diets for long-term success.",
                "Track your food intake and physical activity to stay accountable.",
                "Get enough sleep as poor sleep can affect hunger hormones and weight.",
                "Stay consistent with healthy habits rather than pursuing perfection."
            ],
            
            # Disease Prevention
            "disease_prevention": [
                "Maintain healthy blood pressure through diet, exercise, and stress management.",
                "Control blood sugar levels by limiting refined carbs and sugary foods.",
                "Keep cholesterol in check by reducing saturated and trans fats.",
                "Get vaccinated as recommended and practice good hygiene to prevent infections.",
                "Protect your skin from sun exposure to prevent skin damage and cancer.",
                "Practice safe food handling to prevent foodborne illnesses.",
                "Get regular cancer screenings as recommended for your age and gender."
            ]
        }
        
        self.symptoms_advice = {
            "headache": "Rest in a quiet, dark room. Apply cold compress to forehead. Stay hydrated. Consider over-the-counter pain relievers. Consult doctor if headache is severe, persistent, or accompanied by other symptoms.",
            "fever": "Rest and stay hydrated. Take paracetamol if needed. Use cool compresses. Monitor temperature regularly. See doctor if fever exceeds 103°F, lasts more than 3 days, or is accompanied by rash or difficulty breathing.",
            "cough": "Drink warm liquids like tea with honey. Use steam inhalation. Avoid irritants like smoke. Consider cough drops. Consult doctor if cough lasts more than 3 weeks, produces blood, or causes breathing difficulties.",
            "cold": "Get plenty of rest. Drink fluids like water, juice, and clear broth. Use saline nasal drops. Gargle with salt water. Symptoms usually improve in 7-10 days. See doctor if symptoms worsen or don't improve.",
            "sore throat": "Gargle with warm salt water. Drink warm liquids. Use throat lozenges. Avoid irritants like smoking. Get plenty of rest. See doctor if severe, lasts more than a week, or is accompanied by fever.",
            "stomach pain": "Eat bland foods like bananas, rice, applesauce, toast. Avoid spicy, fatty, or fried foods. Stay hydrated with small sips of water. Rest. See doctor if pain is severe, persistent, or accompanied by fever/vomiting.",
            "diarrhea": "Stay hydrated with oral rehydration solutions. Eat bland foods. Avoid dairy, fatty foods, and caffeine. Rest. See doctor if lasts more than 2 days, has blood, or causes dehydration signs.",
            "constipation": "Increase fiber intake with fruits, vegetables, whole grains. Drink plenty of water. Exercise regularly. Consider fiber supplements if needed. See doctor if persistent or severe.",
            "back pain": "Apply ice or heat. Take over-the-counter pain relievers. Gentle stretching. Maintain good posture. Avoid heavy lifting. See doctor if severe, persistent, or accompanied by leg weakness/numbness.",
            "fatigue": "Ensure adequate sleep. Eat balanced meals. Stay hydrated. Exercise regularly. Manage stress. See doctor if persistent, severe, or affecting daily activities."
        }
        
        self.emergency_keywords = {
            'heart attack': '🚨 EMERGENCY: Chest pain, shortness of breath, or heart attack symptoms require IMMEDIATE medical attention. Call emergency services NOW!',
            'chest pain': '🚨 EMERGENCY: Chest pain can be serious. Please seek IMMEDIATE medical attention or call emergency services!',
            'difficulty breathing': '🚨 EMERGENCY: Breathing difficulties require IMMEDIATE medical care. Call emergency services NOW!',
            'severe bleeding': '🚨 EMERGENCY: For severe bleeding, apply direct pressure and seek IMMEDIATE medical help!',
            'suicide': '🚨 EMERGENCY: Please contact emergency services or a crisis helpline immediately. Your life is valuable!',
            'stroke': '🚨 EMERGENCY: Sudden numbness, confusion, trouble speaking, or vision problems may indicate stroke. Seek IMMEDIATE medical help!',
            'unconscious': '🚨 EMERGENCY: If someone is unconscious, call emergency services immediately!'
        }
        
        self.greetings = [
            "Hello! I'm your health assistant. How can I help you with your health today?",
            "Hi there! Ready to talk about your health and wellness?",
            "Welcome! I'm here to help with your health questions and concerns.",
            "Greetings! I'm your health companion. What would you like to know about your health?"
        ]
        
        self.fallback_responses = [
            "I understand you're asking about health. Could you provide more specific details?",
            "That's an important health topic. Let me help you with more specific information about that.",
            "For detailed medical advice on this topic, it's best to consult a healthcare professional.",
            "I can provide general health information. Could you rephrase your question with more details?",
            "I specialize in general health advice. For specific medical conditions, please consult a doctor."
        ]
        
        # Initialize TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.build_knowledge_base()
    
    def build_knowledge_base(self):
        """Build comprehensive knowledge base for TF-IDF matching"""
        self.documents = []
        self.responses = []
        
        # Add greetings
        for greeting in self.greetings:
            self.documents.append("hello hi hey greeting welcome")
            self.responses.append(greeting)
        
        # Add knowledge from all categories
        for category, info in self.health_knowledge.items():
            for item in info:
                keywords = category + " " + item.lower()
                self.documents.append(keywords)
                self.responses.append(item)
        
        # Add symptoms knowledge
        for symptom, advice in self.symptoms_advice.items():
            keywords = f"symptom {symptom} health medical"
            self.documents.append(keywords)
            self.responses.append(f"**{symptom.replace('_', ' ').title()}**: {advice}")
        
        # Add emergency responses
        for emergency, response in self.emergency_keywords.items():
            self.documents.append(f"emergency {emergency} urgent critical")
            self.responses.append(response)
        
        # Train TF-IDF
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def get_response(self, user_input):
        """Get intelligent response using multiple matching techniques"""
        user_input = user_input.lower().strip()
        
        # Check for empty input
        if not user_input:
            return "Please ask me a health-related question!"
        
        # Check for greetings
        if any(word in user_input for word in ['hello', 'hi', 'hey', 'hola', 'greetings']):
            return random.choice(self.greetings)
        
        # Check for thanks
        if any(word in user_input for word in ['thank', 'thanks', 'thank you']):
            return "You're welcome! I'm glad I could help. Feel free to ask more health questions!"
        
        # Check for emergency situations first
        for emergency, response in self.emergency_keywords.items():
            if emergency in user_input:
                return response
        
        # Check for specific symptoms
        for symptom, advice in self.symptoms_advice.items():
            if symptom in user_input:
                return f"**{symptom.replace('_', ' ').title()}**: {advice}"
        
        # Use TF-IDF for semantic matching
        try:
            input_vec = self.vectorizer.transform([user_input])
            similarities = cosine_similarity(input_vec, self.tfidf_matrix)
            max_sim_idx = similarities.argmax()
            
            if similarities[0, max_sim_idx] > 0.2:  # Reasonable similarity threshold
                return self.responses[max_sim_idx]
        except Exception as e:
            print(f"TF-IDF Error: {e}")
        
        # Category-based fallback
        category_keywords = {
            'diet_nutrition': ['diet', 'food', 'eat', 'nutrition', 'meal', 'protein', 'carb', 'fat', 'vitamin'],
            'exercise_fitness': ['exercise', 'workout', 'fitness', 'gym', 'run', 'walk', 'yoga', 'cardio', 'strength'],
            'mental_health': ['mental', 'stress', 'anxiety', 'depression', 'mood', 'mind', 'meditation', 'therapy'],
            'weight_management': ['bmi', 'weight', 'fat', 'obese', 'overweight', 'underweight', 'loss', 'gain'],
            'general_health': ['health', 'healthy', 'wellness', 'lifestyle', 'prevention', 'checkup', 'doctor'],
            'disease_prevention': ['disease', 'prevent', 'cancer', 'diabetes', 'blood pressure', 'cholesterol']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                return random.choice(self.health_knowledge[category])
        
        # General health fallback
        if any(word in user_input for word in ['health', 'healthy', 'wellness']):
            return random.choice(self.health_knowledge['general_health'])
        
        return random.choice(self.fallback_responses)

# Initialize enhanced chatbot
health_chatbot = HealthChatbot()

def chatbot_response(user_input):
    """Enhanced chatbot response function"""
    return health_chatbot.get_response(user_input)

# ------------------ LOGIN REQUIRED DECORATOR ------------------
from functools import wraps
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please login to continue!")
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return wrapper

# ------------------ ROUTES ------------------
@app.route('/', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fusername = request.form['username']
        femail = request.form['email']
        fpassword = request.form['password']
        fphone = request.form['phone']

        # ✅ Basic Validation
        if not (fusername and femail and fphone and fpassword):
            flash("All fields are required!")
            return redirect(url_for('register'))

        if not fphone.isdigit() or len(fphone) != 10:
            flash("Phone number must contain exactly 10 digits!")
            return redirect(url_for('register'))

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, femail):
            flash("Invalid email format!")
            return redirect(url_for('register'))

        if len(fpassword) < 6 or not any(c.isdigit() for c in fpassword) or not any(c.isalpha() for c in fpassword):
            flash("Password must be at least 6 characters long and contain both letters and numbers!")
            return redirect(url_for('register'))

        existing_user = User.query.filter(
            (User.username == fusername) | (User.email == femail) | (User.phone == fphone)
        ).first()
        if existing_user:
            flash("User already exists!")
            return redirect(url_for('register'))

        new_user = User(username=fusername, email=femail, password=fpassword, phone=fphone)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully! Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        femail = request.form['email']
        fpassword = request.form['password']

        user = User.query.filter_by(email=femail, password=fpassword).first()
        if user:
            session['user_id'] = user.id
            flash("Login successful!")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password!")
            return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully!")
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    username = None
    user_initial = None
    is_admin = False
    
    if user_id:
        user = User.query.get(user_id)
        username = user.username
        user_initial = user.username[0].upper()
        is_admin = user.is_admin
    
    return render_template('dashboard.html', username=username, user_initial=user_initial, is_admin=is_admin)


@app.route('/personhealth', methods=['GET', 'POST'])
@login_required
def personhealth():
    if request.method == 'POST':
        fage = int(request.form.get('age'))
        fgender = request.form.get('gender')
        fweight = float(request.form.get('weight'))
        fheight = float(request.form.get('height')) / 100
        factivity = request.form.get('activity')
        fgoal = request.form.get('goal')

        user_id = session.get('user_id')
        new_info = HealthInfo(user_id=user_id, age=fage, gender=fgender, weight=fweight, height=fheight,
                              activity_level=factivity, goal=fgoal)
        db.session.add(new_info)
        db.session.commit()

        bmi = round(fweight / (fheight ** 2), 2)

        # ✅ BMI Category & Diet Plan
        if bmi < 18.5:
            category = "Underweight"
            diet = [
                "Eat more protein-rich foods like eggs, meat, fish, and lentils.",
                "Include healthy fats from avocado, nuts, seeds, and olive oil.",
                "Drink milk, smoothies, and nutritional shakes regularly.",
                "Eat frequent small meals throughout the day.",
                "Include complex carbs like whole grains and potatoes."
            ]
        elif 18.5 <= bmi < 24.9:
            category = "Normal Weight"
            diet = [
                "Maintain a balanced diet with fruits, vegetables, and lean proteins.",
                "Stay hydrated and exercise regularly for maintenance.",
                "Avoid junk food and processed foods.",
                "Include variety in your diet for all essential nutrients.",
                "Practice portion control and mindful eating."
            ]
        elif 25 <= bmi < 29.9:
            category = "Overweight"
            diet = [
                "Reduce sugar, refined carbs, and processed foods.",
                "Increase cardio activities like jogging, cycling, swimming.",
                "Eat fiber-rich foods and control portion sizes.",
                "Focus on lean proteins and vegetables.",
                "Limit high-calorie beverages and alcohol."
            ]
        else:
            category = "Obese"
            diet = [
                "Adopt a calorie-deficit diet plan under medical supervision.",
                "Focus on low-impact exercises like walking, swimming, yoga.",
                "Avoid fried and high-fat foods completely.",
                "Eat plenty of vegetables and lean proteins.",
                "Consult with healthcare provider for personalized plan."
            ]

        return render_template("result.html", bmi=bmi, category=category, diet=diet)

    return render_template('personhealth.html')


@app.route('/progress')
@login_required
def progress():
    user_id = session.get('user_id')
    user_data = HealthInfo.query.filter_by(user_id=user_id).all()

    if not user_data:
        flash("No progress data found! Please fill the health form first.")
        return redirect(url_for('personhealth'))

    # ✅ Ensure static folder exists
    os.makedirs("static", exist_ok=True)

    # Prepare data
    weights = [info.weight for info in user_data]
    heights = [info.height for info in user_data]
    bmis = [round(w / (h ** 2), 2) for w, h in zip(weights, heights)]
    entries = list(range(1, len(bmis) + 1))

    # ✅ Plot BMI Progress with Color Zones
    import matplotlib.patches as mpatches
    plt.figure(figsize=(10, 6))
    plt.title("Your BMI Progress Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Entry Number", fontweight='bold')
    plt.ylabel("BMI Value", fontweight='bold')

    # Draw BMI categories (background color bands)
    plt.axhspan(0, 18.4, facecolor='#ADD8E6', alpha=0.3, label='Underweight (<18.5)')
    plt.axhspan(18.5, 24.9, facecolor='#90EE90', alpha=0.3, label='Normal (18.5–24.9)')
    plt.axhspan(25, 29.9, facecolor='#FFD580', alpha=0.3, label='Overweight (25–29.9)')
    plt.axhspan(30, 100, facecolor='#FF7F7F', alpha=0.3, label='Obese (30+)')

    # Plot user's BMI line
    plt.plot(entries, bmis, marker='o', color='blue', linewidth=2, label="Your BMI", markersize=8)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Add value annotations on points
    for i, (x, y) in enumerate(zip(entries, bmis)):
        plt.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Add legend
    handles = [
        mpatches.Patch(color='#ADD8E6', label='Underweight (<18.5)'),
        mpatches.Patch(color='#90EE90', label='Normal (18.5–24.9)'),
        mpatches.Patch(color='#FFD580', label='Overweight (25–29.9)'),
        mpatches.Patch(color='#FF7F7F', label='Obese (30+)'),
    ]
    plt.legend(handles=handles, loc="upper left", fontsize=9)

    # Save graph
    img_path = os.path.join("static", f"progress_{user_id}.png")
    plt.savefig(img_path, bbox_inches='tight', dpi=100)
    plt.close()

    # ✅ Table data
    progress_data = [
        {
            "entry": i+1,
            "age": info.age,
            "gender": info.gender,
            "weight": info.weight,
            "height": info.height * 100,  # Convert back to cm for display
            "bmi": round(info.weight / (info.height ** 2), 2),
            "activity": info.activity_level,
            "goal": info.goal
        }
        for i, info in enumerate(user_data)
    ]

    return render_template("progress.html", img_path=img_path, progress_data=progress_data)


@app.route("/chatbot")
@login_required
def chatbot_page():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    try:
        user_msg = request.json["message"]
        bot_reply = chatbot_response(user_msg)
        return jsonify({"reply": bot_reply})
    except Exception as e:
        return jsonify({"reply": "I'm having trouble processing your request. Please try again."})


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')

# ✅ ADD: Secure Admin Routes
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    """Special admin login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # ✅ HARDCODED ADMIN CREDENTIALS (Change these!)
        if username == "adhimansu" and password == "adhimanshu121":
            # Find or create admin user
            admin_user = User.query.filter_by(username=username).first()
            if not admin_user:
                admin_user = User(
                    username=username,
                    email="admin@healthsystem.com",
                    password=password,
                    phone="0000000000",
                    is_admin=True
                )
                db.session.add(admin_user)
                db.session.commit()
            
            session['user_id'] = admin_user.id
            flash("🔓 Admin login successful!")
            return redirect(url_for('admin_dashboard'))
        else:
            flash("❌ Invalid admin credentials!")
    
    return render_template('admin_login.html')

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    """Admin dashboard"""
    total_users = User.query.count()
    total_health_records = HealthInfo.query.count()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_health_records=total_health_records)

@app.route('/admin/view-data')
@admin_required
def admin_view_data():
    """🔒 SECURE: View all database data - Admin only"""
    try:
        users = User.query.all()
        health_data = HealthInfo.query.all()
        
        user_list = []
        for user in users:
            user_list.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'phone': user.phone,
                'is_admin': user.is_admin
            })
        
        health_list = []
        for data in health_data:
            bmi = round(data.weight / (data.height ** 2), 2)
            health_list.append({
                'id': data.id,
                'user_id': data.user_id,
                'age': data.age,
                'gender': data.gender,
                'weight': data.weight,
                'height': data.height,
                'bmi': bmi,
                'activity': data.activity_level,
                'goal': data.goal
            })
        
        return jsonify({
            'total_users': len(user_list),
            'total_health_records': len(health_list),
            'users': user_list,
            'health_data': health_list
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/delete-user/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Delete user (Admin only)"""
    try:
        user = User.query.get(user_id)
        if user:
            # Delete associated health data first
            HealthInfo.query.filter_by(user_id=user_id).delete()
            db.session.delete(user)
            db.session.commit()
            return jsonify({'success': True, 'message': 'User deleted successfully'})
        return jsonify({'success': False, 'message': 'User not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ------------------ MAIN ------------------
if __name__ == "__main__":
    with app.app_context():
        try:
            db.create_all()
            print("✅ Database tables created successfully!")
        except Exception as e:
            print(f"⚠️ Database warning: {e}")
    
    # ✅ FIX: Render.com compatible port binding
    port = int(os.environ.get("PORT", 5005))
    app.run(debug=False, host='0.0.0.0', port=port)