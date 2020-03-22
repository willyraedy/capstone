import smtplib, ssl
from dotenv import load_dotenv
load_dotenv()
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

password = os.environ['GMAIL_PASSWORD']

port = 465  # For SSL

def create_html_for_article(article):
  title, link = article[0]
  confidence_score = article[1]
  return f"""
    <a href={link}>{title}</a>
    <p>Confidence score: {confidence_score}
    <br>
    """

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
  html = f"""\
  <html>
    <body>
      <h1>Here are the top {len(articles)} articles</h1><br>

      {' '.join([create_html_for_article(a) for a in articles])}
    </body>
  </html>
  """

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
