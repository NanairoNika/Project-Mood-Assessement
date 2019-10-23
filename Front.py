from flask import Flask, request, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SubmitField
from wtforms.validators import DataRequired

from preproc import load_obj, cleaning

app = Flask(__name__)
app.debug = True
app.config.from_object('config')




class Search(FlaskForm):
    query = StringField('Введите поиск:', validators=[DataRequired()])
    method = SelectField("Метод: ", choices=[
        ("tfidf", "tfidf"),
        ("bm25", "bm25"),
        ("fasttext", "fasttext"),
        ("elmo", "elmo")])
    submit = SubmitField("Search")


@app.route('/', methods=['GET', 'POST'])
def search():
    user = {'nickname': 'ANONIMUS'}
    form = Search()
    if form.validate_on_submit():
        qu = form.query.data
        met = form.method.data
        return redirect(url_for('query',
                                qu=qu,
                                met=met))
    return render_template("searchpage.html",
                           user=user,
                           form=form)


@app.route('/querypage')
def query():
    user = {'nickname': 'ANONIMUS'}  # выдуманный пользователь
    return render_template("querypage.html",
                           user=user,
                           qu=request.args.get('qu'),
                           met=request.args.get('met'))


if __name__ == "__main__":
    app.run()
