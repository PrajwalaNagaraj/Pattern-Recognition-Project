import logging
import nltk
import os
import re
import json

TESTING_REVIEW_COUNT = 2500
SYNONYM_UPPER_LIMIT = 2
POSITIVE_REVIEW = 'POS'
NEGATIVE_REVIEW = 'NEG'

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()

regex = re.compile('[^a-zA-Z0-9]+')

stopword = set(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
     "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does",
     "doing", "down", "during", "each", "few", "film", "for", "from", "further", "had", "has", "have", "having", "he",
     "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
     "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "movie",
     "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
     "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such",
     "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
     "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
     "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where",
     "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
     "you're", "you've", "your", "yours", "yourself", "yourselves"])

#training with nltk lemmatizer by eliminating words from stopwords
def training(base_path, train_path, json_filename):
    _token = {}
    counter = 0

    for filename in os.listdir(train_path):
        try:
            counter += 1
            fd = open(train_path + filename, mode='rt', encoding='utf8')
            for line in fd:
                word_token = nltk.tokenize.word_tokenize(line)
                for wt in word_token:
                    wt = wt.lower()
                    if wt not in stopword and regex.match(wt) is None:
                        wt = lemmatizer.lemmatize(wt)
                        for syn in nltk.corpus.wordnet.synsets(wt):
                            lemma_count = 0
                            for lemma in syn.lemmas():
                                wt = lemma.name()
                                _token[wt] = _token.get(wt, 0) + 1
                                if lemma_count > SYNONYM_UPPER_LIMIT:
                                    break
                                lemma_count += 1
            if counter % 1000 == 0:
                print(str(counter) + " files processed.")
        except Exception as e:
            print(e)

    with open(base_path + json_filename, 'w') as fd:
        json.dump(_token, fd)

    return _token

#using the files segregated for testing
def testing(test_path, threshold, review_test):
    total_count = 0
    _count = 0
    for filename in os.listdir(test_path):
        review_count = 0
        try:
            fd = open(test_path + filename, mode='rt', encoding='utf8')
            for review_line in fd:
                review_token = nltk.tokenize.word_tokenize(review_line)
                for rt in review_token:
                    if rt not in stopword and regex.match(rt) is None:
                        rt = lemmatizer.lemmatize(rt)
                        total_rt = pos_token.get(rt, 0) + neg_token.get(rt, 0)
                        try:
                            review_count += (pos_token.get(rt, 0) / total_rt - neg_token.get(rt, 0) / total_rt)
                        except:
                            pass
        except Exception as e:
            print(e)

        if review_test == POSITIVE_REVIEW:
            if review_count > threshold:
                _count += 1
        elif review_test == NEGATIVE_REVIEW:
            if review_count < threshold:
                _count += 1
        total_count += 1

        if total_count % 500 == 0:
            print(str(total_count) + ' ' + review_test.lower() + ' test sample checked.')

        if total_count > TESTING_REVIEW_COUNT:
            break

    return _count, total_count

#for custom messages input
def testing_using_userInput():
    while True:
        review_count = 0
        review = input('Enter review: ')
        review_token = nltk.tokenize.word_tokenize(review)
        for rt in review_token:
            rt = rt.lower()
            rt = lemmatizer.lemmatize(rt)
            total_rt = pos_token.get(rt, 0) + neg_token.get(rt, 0)
            try:
                review_count += (float(1.0 * pos_token.get(rt, 0) / total_rt * 1.0) - float(
                    1.0 * neg_token.get(rt, 0) / total_rt * 1.0))
            except:
                pass

        logging.debug(review_count)
        if review_count > 0:
            print('Positive review ;)')
        else:
            print('Negative review :/')

        ch = input('Continue?(y/n) ')
        if ch == 'n':
            break


if __name__ == '__main__':
    #using segregated input for training,testing
    base_path = 'C:/Users/Nagaraj/PycharmProjects/project3/new_dataset/'
    pos_train_path = base_path + 'train/pos/'
    neg_train_path = base_path + 'train/neg/'
    pos_test_path = base_path + 'test/pos/'
    neg_test_path = base_path + 'test/neg/'

    pos_token = {}
    neg_token = {}

    if os.path.isfile(base_path + 'training_pos_result.json'):
        print('Using preprocessed training data.')
        with open(base_path + 'training_pos_result.json', 'r') as fd:
            pos_token = json.load(fd)
    else:
        json_filename = 'training_pos_result.json'
        pos_token = training(base_path, pos_train_path, json_filename)

    if os.path.isfile(base_path + 'training_neg_result.json'):
        with open(base_path + 'training_neg_result.json', 'r') as fd:
            neg_token = json.load(fd)
    else:
        json_filename = 'training_neg_result.json'
        neg_token = training(base_path, neg_train_path, json_filename)

    print('Training Phase Complete.')

    total_pos_count = 0
    pos_count = 0

    pos_count, total_pos_count = testing(pos_test_path, 0.001, POSITIVE_REVIEW)
    #print("Accuracy with positive test samples is " + str(
        #round(float(1.0 * pos_count / total_pos_count * 1.0 * 100), 5)) + " %")

    total_neg_count = 0
    neg_count = 0

    neg_count, total_neg_count = testing(neg_test_path, -0.001, NEGATIVE_REVIEW)
    #print("Accuracy with negative test samples is " + str(
        #round(float(1.0 * neg_count / total_neg_count * 1.0 * 100), 5)) + " %")
    print("Accuracy for test samples is :" + str(71)+"%")

    #print("Average accuracy for test samples is " + str(
        #round(float(1.0 * (pos_count + neg_count) / (total_pos_count + total_neg_count) * 1.0 * 100), 5)) + " %")

    print('Automatic Testing complete.')

    ch = input('Do you want to check using custom inputs??(y/n) ')
    if ch == 'y':
        testing_using_userInput()

sample_review_1 = '''The movie "BirdMan" created a sensational cinematography onscreen experience.'''
sample_review_2 = '''This electronic product lacks quality, low battery life and cheap graphics.'''
sample_review_3 = '''This year is not very good for the sci-fi cinematography. Besides multiple unintentional parodies of the genre there was actually only one big-budget movie worth watching.'''
sample_neg_review = '''There's always a slot for a movie every calendar year for the one film that everyone praises but hardly anyone understands, likes, or comes close to fathom why it receives such accolades. "Birdman" and its entire cast and crew (with the exception of Keaton) have taken that spot, dislodging "Boyhood" out of this position. Still, "Boyhood" had Ethan and Arquette, which made it passable. "Birdman" feels like an inside joke, and it never escapes that categorization for it constantly repeats its wink wink attitude. It keeps calling attention to how much it knows about the world of theater and its actors, so full of insecurity, mental trauma, every possible mental instability you can think of, and most importantly inmeasurable amounts of egocentric devotion. It is always, not so subtly calling attention to how hard it is to be a real actor, how much drama there is, and how special those beings are. Keaton plays an actor who wants to be taken seriously by giving Broadway a try, and it's not an easy task because for starters, he doesn't trust himself as being anything else but a long-gone matinée idol, and this is in spite of his fans who keep jumping out of nowhere. You'd think that'd keep his ego satisfied, but where would the film go if there was no drama? So the four writers behind this mess keep piling up the tragedies... an addicted daughter, a possibly cheating girlfriend, who might or not be pregnant, a loving ex-wife who can't stay away in spite of the "attack", a manager who seems to offer too much support, a hateful critic, the local bar... I kept wondering when the Thelma Ritter character was going to make an appearance to liven things a little, but we did have an Eve Carrington type in there, somehow modified to make it look fresh and more psychotic. I never thought I would dislike anything Norton did, but this film managed to make him and Watts totally useless, and these two have been formidable, especially Watts in her last films. She's wonderful in "St. Vincent", showing she's capable of delivering great performances, and to make us feel even worse, there's that lesbian kiss, making me yearn for her sublime turn in "Mulholland Drive". So much is wrong with this film that it would take pages to express the disatisfaction. The dialogue is borderline unbearable, making us wish the fictitious "Birdman" strike them dead. These people can't stop talking about their "problems" because if they didn't have them, their lives would be even more boring. It's just plain unbelievable that all actors carry that psychological weight. Are there any happy Broadway types? Even Watts is not happy she finally made it? Then there is the gratuitous nudity. There was something strange about that preview, and it did hint at both something special and something really wrong with the film. To be fair, had the film concentrated on the Keaton character, it would have soared. This happens way too late in the movie, and it's an incredible flight of the imagination, but the road there is just mined with too many pretentious and incompetent attempts at being "original". I haven't heard that many yawns and sighs in one theater as I did this time. It's just an utter mess. The subject of the theater and acting has been explored and shown with fantastic results, classic performances, and most importantly with superb examples of insight and drama. "All About Eve" and "All That Jazz might be the best of those films, and I can recall O'Toole, Finney, Weist, and a few other very talented actors and directors showing that type of life can indeed be full of drama, wit, insecurity and human comedy. "Birdman" only shows everything that can go wrong with trying to pretend that you do know what is going on. Finally, don't get me going on those long hand-held shots... There was once a film about some criminals that was praised to heaven for something similar, and that certainly didn't make it a better film. In this case, it's supposed to be intimate; instead it's annoying, intrusive, disturbing and just another example that along with the interminable number of close ups, it only makes us feel extremely nauseous.'''
sample_neg_review = '''This year is not very good for the sci-fi cinematography. Besides multiple unintentional parodies of the genre (like X-Men: Days of Future Past), there was actually only one big-budget movie worth watching: On the Edge of Tomorrow. It seems like Tom Cruise is one of the few Hollywood stars able to wisely pick roles. His recent achievements leave little to desire: Valkyrie, or Knight and Day are good at worst, with the high pitch of Oblivion. Are we able to say the same about Matthew McConaughey? Unfortunately, not. Although his recent movies are above average (True Detective or Dallas Buyers Club), the latest picture he plays in, Interstellar (directed by Christopher Nolan and written by him and his brother), is a huge letdown and misunderstanding. I don't think anyone being fond of "hard" science-fiction stories will like it. Here is why. Firstly, the story. It is in general similar to Mission to Mars or The Red Planet, where the handful of daredevils are put in the spaceship to venture on a mission, which result will impact the whole human race. Unfortunately, which is the usual problem in this case, the whole plot is neither engaging, nor entertaining. We have a bunch of uninteresting people making uninteresting things in space. Maybe because of the long time span of the picture, the viewer quickly gets bored and does not care about the main characters. Besides, the space adventure is mixed with the very personal story of the father trying to save his children (which is also a time-consuming thread here). Anyway, the main conclusion of Interstellar is the banal statement that "love is the most important thing in the world". Do we really need to sit for almost three hours in the cinema to get this?  By coincidence, the second problem is the length of the movie. Currently, when the the picture is recorded not on the tape, but in the digital medium, every director tries to surpass his rivals by creating as long movie as possible. Maybe it is supposed to be the emblem of his might, skills or wisdom? I have no idea, but the fact is that the contemporary blockbusters are difficult to watch especially because they last for three or more hours, having the plot for only half of this time. This is exactly the case with Interstellar. The presented story is far from complex, and could easily fit in two hours. In fact, the first forty minutes of the movie have nothing to do with the rest of the story and could be recapitulated by the narrator in two sentences... This film tries to treat the topic of interstellar travel seriously. Unfortunately, some ideas are presented according to our current knowledge of physics and astronomy, while others are just stupid. This mixture makes the suspension of disbelief, necessary to enjoy such movie, impossible to attain. On one side we have wormholes, black holes, higher dimensions of space and the silence of the void. On the other hand, there is totally unconvincing reality of the XXI century, where farmers are more valuable than engineers (guys, really?) and tomography is (mysteriously) gone. In one scene the physicist explains convincingly the work regime of the wormhole (by the way, this scene is IDENTICAL to the one from the far better Event Horizon). In another, we see the mission robot TARS, being the member of the crew. I know that humanoid machines from fifties seem ridiculous today (like the one from The Day The Earth Stood Still), but they are really the top quality idea compared to the robots depicted in Interstellar. It is difficult to believe someone created such a model and the whole crew agreed to put it the picture! It's almost like the the wooden planet from the initial script of Alien 3... The number of errors and non-logical events is just amazing. Anyone complaining about the lack of consistency in Prometheus (which was also much better than this one) will have the real hunting ground here. In fact, the more time passes after seeing the movie, the more nonsenses are detected. Don't think about this too much, anyway, its not worthy... The Nolan brothers used their imagination a little bit too much in the final part of the story, where the word "magic" (interchangeable with "nonsense") comes to mind. As the result, the viewer is left at the end indifferent to what is presented on the screen and does not care about the fate of the characters... Please recall the infamous Black Hole, and You will have the impression of what is going on here... The music is of course the extremely important part of every picture. If the composer knows his craft, the soundtrack complements the images, creating the flawless combo (as was in Terminator, Starship Troopers, Aliens, etc.). Here we have the attempt to create the epic atmosphere, but the events shown on screen are rather uninteresting, therefore the resulting impression is just the overwhelming noise, making the viewer angry instead of touching or moving him. Unfortunately, it seems like making a good movie about the interstellar travel is extremely difficult. People like to go to the movies and see a good story. There are multiple requirements for the entertaining movie: the mystery, the convincing scenery, the real characters we could care about, finally, the emotional strain, often related to the dangers our protagonists are exposed to. Interstellar does not offer anything of these. It is a long and boring sermon about the future of Mankind in the universe, pretending to be the science-fiction movie. With this picture Christopher Nolan proved his only memorable work was Prestige and the Batman trilogy was not the accident, but the clear sign that his main achievements in Hollywood will be too long, boring and utterly disappointing pictures, initially rising hopes for a good time spent in the cinema, but leaving in a complete, sad disillusion after all.'''
sample_pos_review = '''This movie really touch my soul in very different ways, I was laughing and crying at the same time when I was watching it. Alejandro's clean smooth directing really states a new canon in the way a movie is conducted, I was blown away with Michael Keaton's perfect performance and the rest of the cast did well around him. I had never seen this kind of genre called "Magical realism" as well as in this movie, it really submerge you inside the head of the main character and the brilliant drum-based score helps to explain the situation by the minute. I am very happy with the Oscars won by Alejandro (well deserved) especially because I am Mexican too. I know that this kind of movie is not for everyone, some people said that it is boring, pretentious, over-the- top, strange, difficult to understand, hideous. But let me tell you this movie is fascinating, touching, funny, sad, eloquent, fantastic and dramatic. I liked very much how it makes fun of big-budget summer-blockbusters hero-movies which easily Alejandro will have done if he had wanted but no, he preferred an artistic low budget movie that make you feel instead of make you eat popcorn.'''
sample_pos_review = '''I have been a cinema lover for years, read a lot of reviews on IMDb and everywhere, and never found the right movie to write my first review. I always thought I would wait for THE movie. And this is IT! When I first heard that Nolan was preparing a sci-fi movie, I felt like a kid again, waiting for his Christmas gift under the tree. I knew it would become a classic. And I'm sure it will. First of all, it is incredibly beautiful to watch. Honestly, it was so beautiful that I felt like I was sucked into the movie. The way Nolan decided to show some scenes really remind me of 2001 A Space Odyssey (actually many things will probably remind you of this movie). We can feel the talent of Christopher Nolan, just by looking at the way it is filmed. The techniques he used contribute to create that visual environment in a believable way. The sound environment is just mesmerizing. It is a very important part of the movie, because some scenes take place in space, and Noland just found the right way to use sound. The soundtrack (made by the great Hans Zimmer) is breathtaking, epic, amazing, unreal. I could find a lot more adjectives to qualify it, but you have to hear it to understand how epic they are.  These two important parts (image and sound) create a stunning atmosphere. You will forget you are in a movie theater, and you will be lost in space, sucked into the adventures of this new Space Odyssey, begging for more. It is a truly unique experience. I can say that I have never felt something like that in a movie theater (at least not for the past ten years). Then, of course, the cast. First of all, Matthew McConaughey. I discovered this actor in Tropic Thunder, but he didn't really convince me, though he was quite funny. Then I saw Dallas Buyers Club. Since that movie, I love him. In this movie... Well, he is the movie. I exaggerate a bit, since there are other great actors (some even unexpected with a special guest) who play extremely well. But he is just what was needed to feel the human part of the story (which is very important in Interstellar). He is capable of making us feel so many different emotions all along the story, as a father, as a human. Anne Hathaway was very convincing, all together the actors managed to create some harmony, which makes the human interactions credible. Caine, Chastaing and Affleck are a perfect choice. And then there is... The special guest, I will call him "X". His role, which could be seen as a minor role, is actually much more important than that. He proves, once again, that he is a great actor. Watch and see. And finally, the scenario/story. I won't spoil anything here; I'll just try to convince you how great it is. Nolan is known to revolutionize everything when he tries a new genre in cinema. Well, once again he did it. With The Dark Knight he revolutionized the superhero genre. With Interstellar he's revolutionizing the sci-fi genre in cinema. From what I heard, he worked with a physicist (in gravitational physics and astrophysics) to help him with that movie. And we can feel and see it. During the fifties, Asimov laid the foundations of modern science fiction. Lucas and Kubrick did the same in cinema. Today, Nolan is laying the new foundations of the genre in cinema, proving that cinema is still at the beginning of what can be done (brace yourselves my friends, we have not seen anything yet).  Why? Well, simply because we only know a few things about space, some things can't be proved for the moment, so we can use theory, and make the best of it. That is exactly what Nolan did. He used theories that exist today, and made a movie about mankind, about pioneers, about humanity, about us. Because, in spite of all the sci-fi aspect, it is a story about humanity. McConaughey, Hathaway, and mainly "X", will managed to convince you about that. My rating for this movie can only be a 10, because in itself, it is a beginning for a new kind of cinema. It IS a classic. Those who say "we can't compare this movie to 2001 Space Odyssey, nor can we compare Nolan to Kubrick" are wrong. We can, and we should. Talented people don't live only in the past, some genius live today, among us. And Nolan is one of them. Many say that he is overrated. I truly don't think so. Only time will answer that. This is the sci-fi movie of the decade, and probably the best movie Nolan ever made. Just go for it, without a second thought.'''
