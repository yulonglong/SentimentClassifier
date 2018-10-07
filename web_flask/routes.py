from flask import render_template
from web_flask import app
from web_flask import model
from web_flask.evaluate_form import EvaluateForm
from flask import render_template, flash, redirect, Markup
import re

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def sentimentEvaluate():
    form = EvaluateForm()

    # Upon receiving movie review from form
    if form.validate_on_submit():
        movie_review = form.movie_review.data
        movie_review = re.sub(r'(?:\r\n)+','\r\n', movie_review)

        percentScore, pdf_filepath = model.evaluate(movie_review)

        flash(Markup('<b>Movie Review</b>:<br>{}'.format(movie_review)))
        if (percentScore >= 50.0):
            flash(Markup('<b>Prediction</b>: <span style="color: green;">Positive - {0:.2f}%</span>'.format(percentScore)))
        else:
            flash(Markup('<b>Prediction</b>: <span style="color: red;">Negative - {0:.2f}%</span>'.format(percentScore)))
        flash(Markup('<b>PDF Visualization:</b>: <a href=/pdf/{}>Click here</a>'.format(pdf_filepath)))
        return redirect('/')

    return render_template('evaluate.html', form=form)


# For retrieving attention visualization pdf files
from flask import send_from_directory
import os
@app.route('/pdf/<path:pdf_filename>')
def send_pdf(pdf_filename):
    pdf_path = os.path.join(os.getcwd(), model.base_path)
    return send_from_directory(pdf_path, pdf_filename, as_attachment=True)
