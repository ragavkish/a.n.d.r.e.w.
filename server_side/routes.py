from flask import render_template

def init_routes(app):
    @app.route('/')
    def home():
        return render_template('home.html')
    
    @app.route('/login')
    def login():
        return render_template('login.html')
    
    @app.route('/signup')
    def signup():
        return render_template('signup.html')