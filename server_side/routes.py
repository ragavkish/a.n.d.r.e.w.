from flask import render_template, request, redirect, url_for, flash, session
from flask_mail import Message
from server_side.models import create_user, authenticate_user
import random

def init_routes(app, mail):
    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'POST':
            username = request.form['username']
            name = request.form['name']
            email_id = request.form['email_id']
            password = request.form['password']

            if create_user(username, name, email_id, password):
                flash("Signup successful! Please log in.", "success")
                return redirect(url_for('login'))
            else:
                flash("Signup failed. Try again.", "danger")

        return render_template('signup.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email_id = request.form['email']
            password = request.form['password']

            if authenticate_user(email_id, password):
                flash("Login successful!", "success")
                return redirect(url_for('home'))
            else:
                flash("Invalid credentials. Try again.", "danger")

        return render_template('login.html')

    @app.route('/send_otp', methods=['POST'])
    def send_otp():
        email = request.form['email']
        otp = str(random.randint(100000, 999999))
        session['otp'] = otp

        msg = Message('Your OTP Code', sender=app.config["MAIL_USERNAME"], recipients=[email])
        msg.body = f'Your OTP is: {otp}'
        mail.send(msg)
        flash("OTP sent successfully!", "info")
        return redirect(url_for('verify_otp'))

    @app.route('/verify_otp', methods=['GET', 'POST'])
    def verify_otp():
        if request.method == 'POST':
            user_otp = request.form['otp']
            if user_otp == session.get('otp'):
                flash("OTP Verified Successfully!", "success")
                return redirect(url_for('home'))
            else:
                flash("Invalid OTP. Try again.", "danger")
        return render_template('verify_otp.html')

    @app.route('/google_login/callback')
    def google_login_callback():
        if not google.authorized:
            return redirect(url_for("google.login"))

        resp = google.get("/oauth2/v2/userinfo")
        if resp.ok:
            user_info = resp.json()
            session['email'] = user_info["email"]
            flash(f"Welcome {user_info['email']}!", "success")
            return redirect(url_for('home'))
        flash("Google Login Failed.", "danger")
        return redirect(url_for('login'))
