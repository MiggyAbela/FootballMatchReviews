import tkinter as tk
from tkinter import scrolledtext
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as pt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer


def fileChosen(chosenTeamToReview, reviewsAvailable):
    # Choosing the review to show according to the one chosen by the user
    if chosenTeamToReview == teamsAvailable[0]:
        folderName = "mancity"
    elif chosenTeamToReview == teamsAvailable[1]:
        folderName = "liverpool"
    else:
        folderName = "chelsea"

    filepath = folderName + "/" + reviewsAvailable + ".txt"

    # Open the text file and read its contents
    # Encoding is used to make sure the apostrophes are not changed with any other symbols and the right text is show
    with open(filepath, "r", encoding="utf-8") as file:
        textInFile = file.read()

    return textInFile


def createSummary():
    # Get the selected Team
    chosenTeamToReview = teamOpt.get()
    # Get the selected match review
    chosenFileReview = reviewOpt.get()

    textInFile = fileChosen(chosenTeamToReview, chosenFileReview)

    stopWords = set(stopwords.words("english"))
    words = word_tokenize(textInFile)  # Splitting the text in the text file, separate words, symbols, etc.

    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    sentences = sent_tokenize(textInFile)  # Splitting the text in separate sentences
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if word in sentence.lower():
                    if sentence in sentenceValue:
                        sentenceValue[sentence] += freq
                    else:
                        sentenceValue[sentence] = freq

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    average = int(sumValues / len(sentenceValue))

    summary = ''

    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    #print("Summary: " + summary)

    with open("summary_reports/summary_of_review.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Summary of Review:\n" + summary + "\n\n")
        file.close()

    # Converting the words to lower case
    summary = summary.lower()

    # Removing any punctuation or digits from text
    summary = re.sub(r'[^\w\s]', '', summary)
    summary = re.sub(r'\d+', '', summary)

    # Tokenizing the text from the chosen text file
    summaryToken = word_tokenize(summary)
    #print(summaryToken)  # testing

    # Appending words tokenized words into file
    with open("summary_reports/summary_tokenization.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Tokenized Summary:\n" + str(summaryToken) + "\n\n")
        file.close()

    # Removing Stopwords from list of tokens
    cleanSummaryTokens = summaryToken[:]
    for token in summaryToken:
        if token in stopwords.words('english'):
            cleanSummaryTokens.remove(token)

    #print(cleanSummaryTokens)  # testing
    # Appending words without stopwords into file
    with open("summary_reports/summary_stopword_removal.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Summary Clean Tokens - No Stopwords:\n" + str(cleanSummaryTokens) + "\n\n")
        file.close()

    # Creating a Tfidf instance
    tfidf = TfidfVectorizer()

    # Trasforming the cleanSummaryTokens into a Tfidf matrix
    tfidfMatrix = tfidf.fit_transform(cleanSummaryTokens)

    # Obtain sum of Tfidf values from the match review
    sumValue = tfidfMatrix.sum(axis=0)

    # Getting word and sum respectively
    scoresForWords = [(word, sumValue[0, idx]) for word, idx in tfidf.vocabulary_.items()]

    # Sorting the list in descending order based on the Tfidf score
    topTenWords = sorted(scoresForWords, key=lambda x: x[1], reverse=True)[:10]

    with open("summary_reports/summary_tfidf_score.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Summary Tfidf Score:\n")
        for word, score in topTenWords:
            file.write(str(word) + ": " + str(score) + "\n")
        file.write("\n\n")
        file.close()


def updateReviewTxtArea(*args):
    # Get the selected Team
    chosenTeamToReview = teamOpt.get()
    # Get the selected match review
    chosenFileReview = reviewOpt.get()

    textInFile = fileChosen(chosenTeamToReview, chosenFileReview)

    # Updating the text in the non-editable text area
    reviewTxtArea.config(state=tk.NORMAL)
    reviewTxtArea.delete("1.0", tk.END)
    reviewTxtArea.insert(tk.END, textInFile)
    reviewTxtArea.config(state=tk.DISABLED)

    # Converting the words to lower case
    textInFile = textInFile.lower()

    # Removing any punctuation or digits from text
    textInFile = re.sub(r'[^\w\s]', '', textInFile)
    textInFile = re.sub(r'\d+', '', textInFile)

    # Tokenizing the text from the chosen text file
    tokens = word_tokenize(textInFile)
    #print(tokens) #testing

    # Appending words tokenized words into file
    with open("review_reports/review_tokenization.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Tokenized Review:\n" + str(tokens) + "\n\n")
        file.close()


    # Removing Stopwords from list of tokens
    cleanTokens = tokens[:]
    for token in tokens:
        if token in stopwords.words('english'):
            cleanTokens.remove(token)
    #print(cleanTokens) # testing

    # Appending words without stopwords into file
    with open("review_reports/review_stopword_removal.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Clean Tokens - No Stopwords:\n" + str(cleanTokens) + "\n\n")
        file.close()

    # Lemming the cleanedTokens
    lemmatizer = WordNetLemmatizer()
    lemmatizedTokens = []
    for word in cleanTokens:
        lemmatizedText = lemmatizer.lemmatize(word)
        lemmatizedTokens.append(lemmatizedText)
    # testing
    # for x in lemmatizedTokens:
    #     print(x)

    # Appending Lemmatized tokens from review to file
    with open("review_reports/review_lemmatization.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Lemmatized Tokens:\n" + str(lemmatizedTokens) + "\n\n")
        file.close()

    # Conducting Sentiment Analysis
    sa = SentimentIntensityAnalyzer()

    positivetxt = 0
    negativetxt = 0
    neutraltxt = 0

    for i in lemmatizedTokens:
        #print(i) # testing
        sentimentScore = sa.polarity_scores(i)
        score = sentimentScore['compound']

        polarity = ""
        if score > 0:
            polarity = "Positive"
            positivetxt += 1
        elif score < 0:
            polarity = "Negative"
            negativetxt += 1
        else:
            polarity = "Neutral"
            neutraltxt += 1

    sentimentScoreLbl.config(text="Team Performance Rating: " + polarity)

    # Adding details to .txt File
    with open("review_reports/game_polarity_details.txt", "a") as file:
        file.write(str(chosenTeamToReview) + " " + str(chosenFileReview) + " Polarity: " + polarity + "\n")
        file.close()

    # Creating a graph based on the frequencies of each Lemmatized Token
    review_token_freq = nltk.FreqDist(lemmatizedTokens)
    # Setting the size of the graph tab
    pt.figure(figsize=(10, 5))
    review_token_freq.plot(20, cumulative=False)

    createSummary()


def chosenReviewUpdate(*args):
    # Get the selected Team
    chosenTeamToReview = teamOpt.get()
    # Get the selected match review
    chosenFileReview = reviewOpt.get()

    # Update the text of chosenReviewLbl to show the selected option from the review drop-down menu
    chosenReviewLbl.config(text="Match Review: " + chosenTeamToReview + " - " + chosenFileReview)


# Defining the options for the respective drop-down lists
teamsAvailable= ["Manchester City", "Liverpool", "Chelsea"]
reviewsAvailable = ["Game 1", "Game 2", "Game 3", "Game 4", "Game 5"]

# Creating the size and title of the tab
window = tk.Tk()
window.geometry("900x700")
window.title("Football Match Review")

# Drop-down menu for the user to choose a team
teamLbl = tk.Label(window, text="Choose a Team:")
teamLbl.pack(pady=15)
teamOpt = tk.StringVar(value=teamsAvailable[0])
# The '*' is used to unpack the list of teams
teamMenu = tk.OptionMenu(window, teamOpt, *teamsAvailable)
teamMenu.pack()

# Drop-down menu for the user to choose a match review depending on the team
reviewLbl = tk.Label(window, text="Select a Review:")
reviewLbl.pack(pady=15)
reviewOpt = tk.StringVar(value=reviewsAvailable[0])
# The '*' is used to unpack the list of reviews
reviewMenu = tk.OptionMenu(window, reviewOpt, *reviewsAvailable)
reviewMenu.pack()

# The non-editable text area to show the chosen match review
chosenReviewLbl = tk.Label(window, text="Match Review: ")
chosenReviewLbl.pack(pady=15)
reviewTxtArea = scrolledtext.ScrolledText(window, state="disabled", width=90, font=("Helvetica", 12))
reviewTxtArea.pack()

# The label at the bottom of the screen
sentimentScoreLbl = tk.Label(window, text="Team Performance Rating:")
sentimentScoreLbl.pack(pady=15)

# 'w' tells the program to write
reviewOpt.trace("w", chosenReviewUpdate)

# Update text in text area
reviewOpt.trace("w", updateReviewTxtArea)

# Get the selected Team
chosenTeamToReview = teamOpt.get()

# Get the selected match review
chosenFileReview = reviewOpt.get()

# mainloop is a built-in method used to start the event
window.mainloop()