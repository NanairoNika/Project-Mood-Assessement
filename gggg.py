from flask import Flask, request, current_app, render_template
app = Flask(__name__)
app.debug=True

@app.route('/id')
def main():
    return render_template('base.html')

if __name__ == "__main__":
    app.run()