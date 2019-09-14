# Telegram Bot with Keras
Using Telegram Bot API and Xception model for predicting sent user's picture

# Create bot
First of all you need to register your bot in [BotFather chat](https://telegram.me/botfather). Follow the instructions that bot say and your token. Which allow you to get access the HTTP API for your application.
### Token
You have to create an `Updater` object. Replace `'TOKEN'` with your Bot's API token. For additional details about [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot).
```python
updater = Updater(token='TOKEN', use_context=True)
```

# Getting started
To start conversation with bot write in the chat field `/start`(When you first start it will be automatically).

Then bot ask you to send an image. After that bot downloading the image. Then user need to press `Recognize` to start predicting what is shown in the pictue. And finally bot printing top1-accuracy in the chat.

### Here is example
![image](https://user-images.githubusercontent.com/43681334/64707626-efe13f00-d4bb-11e9-82c8-d8617808c505.png)
