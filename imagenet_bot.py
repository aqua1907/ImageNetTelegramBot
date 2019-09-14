import logging

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)
import tensorflow as tf
from keras.backend import tensorflow_backend
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

import threading

# Setup gpu config for tensorflow
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


class ImageNetBot:
    def __init__(self, token):
        # Enable logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.token = token

        self.updater = Updater(self.token, use_context=True)

        self.PICTURE, self.PREDICTION = range(2)

        self.keyboard = [['Recognize', '/cancel', 'Stop the bot']]
        self.reply_markup = ReplyKeyboardMarkup(self.keyboard, resize_keyboard=True, one_time_keyboard=True)

    def start(self, update, context):
        """
        Bot's starting method which greetings user
        """
        update.message.reply_text('Send me an image!', reply_markup=self.reply_markup)

        return self.PICTURE

    def picture(self, update, context):
        """
        Grab user's picture and dowload to bot's host storage
        returns PREDICTION that means switch to prediction() method
        """
        user = update.message.from_user
        id = str(user.id)
        photo_file = update.message.photo[-1].get_file()
        photo_file.download('userID_{}_photo.jpg'.format(id))
        update.message.reply_text('[INFO] Downloading the image... ')
        self.logger.info("Image of %s: %s", user.first_name, 'userID_{}_photo.jpg'.format(id))

        return self.PREDICTION

    def prediction(self, update, context):
        """
        Prediction method that using pre-trained on ImageNet dataset
        Xception model for recognizing sent user's picture
        returns PICTURE that means switch to picture() method
        you can infinitely send images to bot until you print /cancel 
        or kill bot process by
        """
        
        id = str(update.message.from_user.id)
        img = image.load_img('userID_{}_photo.jpg'.format(id), target_size=(299, 299))

        update.message.reply_text('[INFO] Preprocessing...')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        update.message.reply_text('[INFO] Recognizing...')
        model = Xception()
        preds = model.predict(x)

        decoded_preds = decode_predictions(preds, top=1)[0][0]

        update.message.reply_text('Predicted: {} with {:.2f}% accuracy'.format(decoded_preds[1],
                                                                               decoded_preds[2] * 100))

        update.message.reply_text('Send me another image or /cancel for stop conversation')

        return self.PICTURE

    def cancel(self, update, context):
        """
        Method that stops the conversation between user and bot
        """
        user = update.message.from_user
        self.logger.info("User %s canceled the conversation.", user.first_name)
        update.message.reply_text('Bye! Text me /start for new session',
                                  reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def error(self, update, context):
        """
        Method for printing errors by using logger
        """
        """Log Errors caused by Updates."""
        self.logger.warning('Update "%s" caused error "%s"', update, context.error)

    def shutdown(self):
        """
        Method for stop bot's process
        """
        self.updater.stop()
        self.updater.is_idle = False

    def stop(self, bot, update):
        """ 
        By using threading activate shutdown() method
        for kill bot's process
        """
        threading.Thread(target=self.shutdown).start()

    def main(self):
        # Create the Updater and pass it your bot's token.
        # Make sure to set use_context=True to use the new context based callbacks
        # Post version 12 this will no longer be necessary

        # Get the dispatcher to register handlers
        dp = self.updater.dispatcher

        # Add conversation handler with the states PICTURE, PREDICTION
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start',  self.start)],

            states={

                self.PICTURE: [MessageHandler(Filters.photo,  self.picture)],
                self.PREDICTION: [MessageHandler(Filters.text,  self.prediction)]
            },

            fallbacks=[CommandHandler('cancel', self.cancel),
                       MessageHandler(Filters.regex('^Stop the bot$'), self.stop)]
        )

        dp.add_handler(conv_handler)

        # log all errors
        dp.add_error_handler(self.error)

        # Start the Bot
        self.updater.start_polling()


if __name__ == '__main__':
    # Insert telegram bot api token
    ImgNet = ImageNetBot("")
    ImgNet.main()