import smtplib, ssl

port = 587  # For starttls
smtp_server = "smtp.gmail.com"
sender_email = "vikas2dx@gmail.com"
receiver_email = "vikas2dx@gmail.com"
password = "coolvikas123!!!"
message = """\
Subject: Hi there

This message is sent from Python."""


def sendEmail(name):
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.ehlo()  # Can be omitted
        server.starttls(context=context)
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, name+ " has not weared mask")

sendEmail("vikas")
