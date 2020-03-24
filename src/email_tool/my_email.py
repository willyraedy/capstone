import smtplib, ssl
from dotenv import load_dotenv
load_dotenv()
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Environment, FileSystemLoader

password = os.environ['GMAIL_PASSWORD']

port = 465  # For SSL

def render_email_html(**kwargs):
  root = os.path.dirname(os.path.abspath(__file__))
  templates_dir = os.path.join(root, 'templates')
  env = Environment( loader = FileSystemLoader(templates_dir) )
  template = env.get_template('email.html')

  return template.render(**kwargs)

def send_climate_articles(receiver_email, articles):
  sender_email = "wilburdad84637@gmail.com"

  message = MIMEMultipart("alternative")
  message["Subject"] = "Climate Articles for Reddit"
  message["From"] = sender_email
  message["To"] = receiver_email

  # Create the plain-text and HTML version of your message
  text = """\
  Hi,
  How are you?
  Real Python has many great tutorials:
  www.realpython.com"""
  date_string = '1/1/2020'
  html = render_email_html(articles=articles, date=date_string)

  # Turn these into plain/html MIMEText objects
  part1 = MIMEText(text, "plain")
  part2 = MIMEText(html, "html")

  # Add HTML/plain-text parts to MIMEMultipart message
  # The email client will try to render the last part first
  message.attach(part1)
  message.attach(part2)

  # Create a secure SSL context
  context = ssl.create_default_context()

  # Connect with sending gmail server
  with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
      server.login(sender_email, password)
      server.sendmail(
          sender_email, receiver_email, message.as_string()
      )

