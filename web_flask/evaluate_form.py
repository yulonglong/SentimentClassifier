from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired

class EvaluateForm(FlaskForm):
    movie_review = TextAreaField('Free-text Movie Review:', validators=[DataRequired()])
    submit = SubmitField('Submit')
