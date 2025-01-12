from flask import render_template, request, redirect, url_for, flash
from server_side.models import create_user, authenticate_user

def init_routes(app):
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
