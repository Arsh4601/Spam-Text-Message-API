from flask import Flask,jsonify,render_template
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob
import joblib
import re

port_stem=PorterStemmer()

app = Flask(__name__)

def stemming(content):
    
    stemming_content=re.sub("[^a-zA-Z]"," ",content)
    stemming_content=stemming_content.lower()
    visit=content.find("visit")
    if visit!=1:
        
        fmsg=content[:visit]+"https:"+content[visit+5:]
        fmsg.replace("/","")
        stemming_content=fmsg

    stemming_content=stemming_content.split()
    stemming_content=[port_stem.stem(word) for word in stemming_content if not word in stopwords.words("english")]
    stemming_content=" ".join(stemming_content)
    
    return stemming_content

tfidf=joblib.load("C:/Users/Arsh/Desktop/my project/spam text-classification/vectorizer.pkl")
mnb=joblib.load("C:/Users/Arsh/Desktop/my project/spam text-classification/mnb_classifier.pkl")

@app.route("/")

def home():
    return render_template("about.html")


@app.route("/isspam/<string:msg>")

def is_spam(msg):

    stem_content=stemming(msg)

    corpus=[stem_content]

    x=tfidf.transform(corpus)

    y_pred=mnb.predict(x)

    res=y_pred[0]

    if res==0:

        res="Real"
    
    else:

        res="Spam"
    
    subject=TextBlob(msg).sentiment.subjectivity
    polarity=TextBlob(msg).sentiment.polarity

    if polarity>0:

        polarity="Positive"
    
    elif  polarity==0:

        polarity="Neutral"

    elif  polarity<0:

        polarity="Negative"    
    
   
    result={"Message":msg,"Is_spam":res,"sentiment":polarity,"subjectivity":subject}

    return jsonify(result)



if __name__=="__main__":

    app.run(debug=True)