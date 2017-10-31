# WhatChat

This is the code I wrote to build `wotchat` (www.wotchat.com)

`wotChat` analyzes WhatsApp® chats and applies NLP to extract meaningful information.

It can answer questions like:

- Who is the funny one in your group? 
- Who is the neede one? 

The code base is hosted on AWS, where a lambda service received the email with the chat in attachement, process it, and updates database and website that displays the output. 
The user finally receives the email with the link that points to the website (results are shown on the browser using d3.js).

From a user perspective, it works in three simple steps:

1. On your phone, open a chat from WhatsApp®
2. Click on "Export Chat", "Without Media"
3. Send to send@wotchat.com as email

An email will be received soon!


Stefano Cosentino.
