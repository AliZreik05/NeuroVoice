from datetime import datetime, timedelta
import os, secrets, re
from email_validator import validate_email, EmailNotValidError
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from flask_mail import Mail, Message

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_DEFAULT_SENDER", app.config["MAIL_USERNAME"])

mail = Mail(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

# ---------- Models ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

class OTPCode(db.Model):
    """Stores one-time codes for actions like password reset."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    code = db.Column(db.String(12), nullable=False)  # 6-digit numeric or short token
    purpose = db.Column(db.String(32), nullable=False)  # e.g., "reset"
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)

    user = db.relationship("User", backref="otps")

# ---------- Auth plumbing ----------
@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))


def send_email(to_email: str, subject: str, body: str):
    """
    Sends an email via the configured SMTP server.
    """
    try:
        msg = Message(sender=('NeuroVoice', f'{app.config["MAIL_DEFAULT_SENDER"]}'), subject=subject, recipients=[to_email], body=body)
        mail.send(msg)
        print(f"[EMAIL SENT] To: {to_email}")
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")


def valid_password(pw: str) -> bool:
    return len(pw) >= 8

def generate_numeric_code(n=6) -> str:
    return "".join(secrets.choice("0123456789") for _ in range(n))

# ---------- Routes ----------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return f"Hello, {current_user.email}! <a href='{url_for('logout')}'>Logout</a>"
    return "Hello! <a href='/login'>Login</a> | <a href='/register'>Register</a> | <a href='/reset'>Reset Password</a>"

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirmPassword = request.form.get("confirmPassword", "")

        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(str(e), "danger")
            return redirect(url_for("register"))

        if password != confirmPassword:
            flash("Passwords do not match.")
            return redirect(url_for("register"))

        if not valid_password(password):
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))

        if User.query.filter_by(username=username).first():
            flash("Username already used.", "warning")
            return redirect(url_for("register"))

        user = User(email=email, username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registered! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template('Register.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid credentials.", "danger")
            return redirect(url_for("login"))
        login_user(user)
        flash("Logged in!", "success")
        return redirect(url_for("inputPage"))

    return render_template('LoginPage.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

@app.route("/reset", methods=["GET", "POST"])
def reset():
    """
    Step 1: user enters email; we send a code and store it in OTPCode with expiry.
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        try:
            validate_email(email)
        except EmailNotValidError as e:
            flash(str(e), "danger")
            return redirect(url_for("reset"))

        user = User.query.filter_by(email=email).first()
        if not user:
            # Don't reveal whether email exists; still say "sent"
            flash("If the email exists, a code was sent.", "info")
            return redirect(url_for("verify_code"))

        # invalidate previous unused reset codes
        OTPCode.query.filter_by(user_id=user.id, purpose="reset", used=False).delete()

        code = generate_numeric_code(6)
        otp = OTPCode(
            user_id=user.id,
            code=code,
            purpose="reset",
            expires_at=datetime.utcnow() + timedelta(minutes=10)
        )
        db.session.add(otp)
        db.session.commit()

        send_email(
            to_email=user.email,
            subject="Your password reset code",
            body=f"Use this code to reset your password: {code}\nIt expires in 10 minutes."
        )

        # store the email in session to link the flow
        session["reset_email"] = user.email
        flash("If the email exists, a code was sent.", "info")
        return redirect(url_for("verify_code"))

    return render_template('ResetPassword.html')

@app.route("/verify-code", methods=["GET", "POST"])
def verify_code():
    """
    Step 2: user enters the 6-digit code.
    If valid, mark it used (or keep reserved) and allow setting new password.
    """
    email = session.get("reset_email")
    if request.method == "POST":
        code = re.sub(r"\s+", "", request.form.get("code", ""))
        email = session.get("reset_email")
        if not email:
            flash("Reset session expired. Start again.", "warning")
            return redirect(url_for("reset"))

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Reset session expired. Start again.", "warning")
            return redirect(url_for("reset"))

        otp = OTPCode.query.filter_by(user_id=user.id, purpose="reset", used=False, code=code).first()
        if not otp or otp.expires_at < datetime.utcnow():
            flash("Invalid or expired code.", "danger")
            return redirect(url_for("verify_code"))

        # Mark code as used and allow setting password
        otp.used = True
        db.session.commit()
        session["reset_allowed_for_user_id"] = user.id
        flash("Code verified. You can now set a new password.", "success")
        return redirect(url_for("set_new_password"))

    return render_template('VerifyCode.html')

@app.route("/set-new-password", methods=["GET", "POST"])
def set_new_password():
    """
    Step 3: set a new password after code verification.
    """
    uid = session.get("reset_allowed_for_user_id")
    if not uid:
        flash("You need to verify a reset code first.", "warning")
        return redirect(url_for("reset"))

    user = User.query.get(uid)
    if not user:
        flash("Session invalid.", "danger")
        return redirect(url_for("reset"))

    if request.method == "POST":
        pw1 = request.form.get("password", "")
        pw2 = request.form.get("password2", "")
        if pw1 != pw2:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("set_new_password"))
        if not valid_password(pw1):
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("set_new_password"))

        user.set_password(pw1)
        db.session.commit()
        # Clear session flags
        session.pop("reset_allowed_for_user_id", None)
        session.pop("reset_email", None)

        flash("Password updated. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template('ResetVerify.html', email=user.email)


@app.route("/input")
def inputPage():
    if current_user.is_authenticated:
        return render_template('AIPage.html')
    return redirect(url_for('login'))


# ---------- CLI helper ----------
@app.cli.command("init-db")
def init_db():
    """flask init-db"""
    db.create_all()
    print("DB initialized.")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=False)
