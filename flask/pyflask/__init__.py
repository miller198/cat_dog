import os, sys
real_path = os.path.dirname(os.path.realpath(__file__))
sub_path = os.path.split(real_path)[0]
os.chdir(sub_path)

from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from . import classifier

app = Flask(__name__)
app.debug = True

def root_path():
	real_path = os.path.dirname(os.path.realpath(__file__))
	# sub_path = "/".join(real_path.split("/")[:-1])
	return os.chdir(real_path)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/post', methods=['GET','POST'])
def post():
	if request.method == 'POST':
		root_path()
		# Reference Image
		# refer_img = request.form['refer_img']
		# refer_img_path = './static/images/ref/'+str(refer_img)

		# User Image (target image)
		user_img = request.files['user_img']
		user_img.save('./static/images/usr/'+str(user_img.filename))
		user_img_path = 'images/usr/'+str(user_img.filename)

		# Neural Style Transfer 
		img_result = classifier.main(user_img_path)
		# transfer_img_path = 'images/'+str(transfer_img.split('/')[-1])

	return render_template('post.html', 
					user_img=user_img_path, predict_result=img_result)