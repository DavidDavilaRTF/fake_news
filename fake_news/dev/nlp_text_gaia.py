import pandas
import numpy
import nltk
import csv
csv.field_size_limit(100000000)

def k_to_int(x):
    if x.find('k') != -1:
        return float(x.replace('k','')) * 1000
    return float(x)

def word_listing(x):
    x = x.split('/')
    x = x[3]
    x = x.split('.')
    x = x[0]
    return x.split('-')

def is_words(x,words):
    x = x.lower()
    return w in x

import sys
try:
    arguments = sys.argv
    nb_worker = int(arguments[1])
    worker_id = int(arguments[2])
    nb_feat =  int(arguments[3])
except:
    nb_worker = 5
    worker_id = 0
    nb_feat =  1000
    
gaia = pandas.read_csv('C:/fake_news/doc/note_text.csv',sep = ';',engine = 'python',encoding='utf-8')
sel = numpy.array(gaia['note'].apply(lambda x: str(x).lower() == 'nan'))
sel = sel == False
gaia = gaia[sel]
gaia.index = range(len(gaia))
sel = numpy.array(gaia['text'].apply(lambda x: str(x).lower() == 'nan'))
sel = sel == False
gaia = gaia[sel]
gaia.index = range(len(gaia))
gaia['note'] = gaia['note'].apply(lambda x:k_to_int(str(x)))
words = []
for w in gaia['text']:
    words += nltk.tokenize.word_tokenize(str(w).lower())

words = pandas.DataFrame(words)
words = words[0].unique()

stop_words = nltk.corpus.stopwords.words('french')
stop_words += ['''a''','''abord''','''absolument''','''afin''','''ah''','''ai''','''aie''','''aient''','''aies''','''ailleurs''','''ainsi''','''ait''','''allaient''','''allo''','''allons''','''allô''','''alors''','''anterieur''','''anterieure''','''anterieures''','''apres''','''après''','''as''','''assez''','''attendu''','''au''','''aucun''','''aucune''','''aucuns''','''aujourd''','''aujourd'hui''','''aupres''','''auquel''','''aura''','''aurai''','''auraient''','''aurais''','''aurait''','''auras''','''aurez''','''auriez''','''aurions''','''aurons''','''auront''','''aussi''','''autant''','''autre''','''autrefois''','''autrement''','''autres''','''autrui''','''aux''','''auxquelles''','''auxquels''','''avaient''','''avais''','''avait''','''avant''','''avec''','''avez''','''aviez''','''avions''','''avoir''','''avons''','''ayant''','''ayez''','''ayons''','''b''','''bah''','''bas''','''basee''','''bat''','''beau''','''beaucoup''','''bien''','''bigre''','''bon''','''boum''','''bravo''','''brrr''','''c''','''car''','''ce''','''ceci''','''cela''','''celle''','''celle-ci''','''celle-là''','''celles''','''celles-ci''','''celles-là''','''celui''','''celui-ci''','''celui-là''','''celà''','''cent''','''cependant''','''certain''','''certaine''','''certaines''','''certains''','''certes''','''ces''','''cet''','''cette''','''ceux''','''ceux-ci''','''ceux-là''','''chacun''','''chacune''','''chaque''','''cher''','''chers''','''chez''','''chiche''','''chut''','''chère''','''chères''','''ci''','''cinq''','''cinquantaine''','''cinquante''','''cinquantième''','''cinquième''','''clac''','''clic''','''combien''','''comme''','''comment''','''comparable''','''comparables''','''compris''','''concernant''','''contre''','''couic''','''crac''','''d''','''da''','''dans''','''de''','''debout''','''dedans''','''dehors''','''deja''','''delà''','''depuis''','''dernier''','''derniere''','''derriere''','''derrière''','''des''','''desormais''','''desquelles''','''desquels''','''dessous''','''dessus''','''deux''','''deuxième''','''deuxièmement''','''devant''','''devers''','''devra''','''devrait''','''different''','''differentes''','''differents''','''différent''','''différente''','''différentes''','''différents''','''dire''','''directe''','''directement''','''dit''','''dite''','''dits''','''divers''','''diverse''','''diverses''','''dix''','''dix-huit''','''dix-neuf''','''dix-sept''','''dixième''','''doit''','''doivent''','''donc''','''dont''','''dos''','''douze''','''douzième''','''dring''','''droite''','''du''','''duquel''','''durant''','''dès''','''début''','''désormais''','''e''','''effet''','''egale''','''egalement''','''egales''','''eh''','''elle''','''elle-même''','''elles''','''elles-mêmes''','''en''','''encore''','''enfin''','''entre''','''envers''','''environ''','''es''','''essai''','''est''','''et''','''etant''','''etc''','''etre''','''eu''','''eue''','''eues''','''euh''','''eurent''','''eus''','''eusse''','''eussent''','''eusses''','''eussiez''','''eussions''','''eut''','''eux''','''eux-mêmes''','''exactement''','''excepté''','''extenso''','''exterieur''','''eûmes''','''eût''','''eûtes''','''f''','''fais''','''faisaient''','''faisant''','''fait''','''faites''','''façon''','''feront''','''fi''','''flac''','''floc''','''fois''','''font''','''force''','''furent''','''fus''','''fusse''','''fussent''','''fusses''','''fussiez''','''fussions''','''fut''','''fûmes''','''fût''','''fûtes''','''g''','''gens''','''h''','''ha''','''haut''','''hein''','''hem''','''hep''','''hi''','''ho''','''holà''','''hop''','''hormis''','''hors''','''hou''','''houp''','''hue''','''hui''','''huit''','''huitième''','''hum''','''hurrah''','''hé''','''hélas''','''i''','''ici''','''il''','''ils''','''importe''','''j''','''je''','''jusqu''','''jusque''','''juste''','''k''','''l''','''la''','''laisser''','''laquelle''','''las''','''le''','''lequel''','''les''','''lesquelles''','''lesquels''','''leur''','''leurs''','''longtemps''','''lors''','''lorsque''','''lui''','''lui-meme''','''lui-même''','''là''','''lès''','''m''','''ma''','''maint''','''maintenant''','''mais''','''malgre''','''malgré''','''maximale''','''me''','''meme''','''memes''','''merci''','''mes''','''mien''','''mienne''','''miennes''','''miens''','''mille''','''mince''','''mine''','''minimale''','''moi''','''moi-meme''','''moi-même''','''moindres''','''moins''','''mon''','''mot''','''moyennant''','''multiple''','''multiples''','''même''','''mêmes''','''n''','''na''','''naturel''','''naturelle''','''naturelles''','''ne''','''neanmoins''','''necessaire''','''necessairement''','''neuf''','''neuvième''','''ni''','''nombreuses''','''nombreux''','''nommés''','''non''','''nos''','''notamment''','''notre''','''nous''','''nous-mêmes''','''nouveau''','''nouveaux''','''nul''','''néanmoins''','''nôtre''','''nôtres''','''o''','''oh''','''ohé''','''ollé''','''olé''','''on''','''ont''','''onze''','''onzième''','''ore''','''ou''','''ouf''','''ouias''','''oust''','''ouste''','''outre''','''ouvert''','''ouverte''','''ouverts''','''o|''','''où''','''p''','''paf''','''pan''','''par''','''parce''','''parfois''','''parle''','''parlent''','''parler''','''parmi''','''parole''','''parseme''','''partant''','''particulier''','''particulière''','''particulièrement''','''pas''','''passé''','''pendant''','''pense''','''permet''','''personne''','''personnes''','''peu''','''peut''','''peuvent''','''peux''','''pff''','''pfft''','''pfut''','''pif''','''pire''','''pièce''','''plein''','''plouf''','''plupart''','''plus''','''plusieurs''','''plutôt''','''possessif''','''possessifs''','''possible''','''possibles''','''pouah''','''pour''','''pourquoi''','''pourrais''','''pourrait''','''pouvait''','''prealable''','''precisement''','''premier''','''première''','''premièrement''','''pres''','''probable''','''probante''','''procedant''','''proche''','''près''','''psitt''','''pu''','''puis''','''puisque''','''pur''','''pure''','''q''','''qu''','''quand''','''quant''','''quant-à-soi''','''quanta''','''quarante''','''quatorze''','''quatre''','''quatre-vingt''','''quatrième''','''quatrièmement''','''que''','''quel''','''quelconque''','''quelle''','''quelles''','''quelqu'un''','''quelque''','''quelques''','''quels''','''qui''','''quiconque''','''quinze''','''quoi''','''quoique''','''r''','''rare''','''rarement''','''rares''','''relative''','''relativement''','''remarquable''','''rend''','''rendre''','''restant''','''reste''','''restent''','''restrictif''','''retour''','''revoici''','''revoilà''','''rien''','''s''','''sa''','''sacrebleu''','''sait''','''sans''','''sapristi''','''sauf''','''se''','''sein''','''seize''','''selon''','''semblable''','''semblaient''','''semble''','''semblent''','''sent''','''sept''','''septième''','''sera''','''serai''','''seraient''','''serais''','''serait''','''seras''','''serez''','''seriez''','''serions''','''serons''','''seront''','''ses''','''seul''','''seule''','''seulement''','''si''','''sien''','''sienne''','''siennes''','''siens''','''sinon''','''six''','''sixième''','''soi''','''soi-même''','''soient''','''sois''','''soit''','''soixante''','''sommes''','''son''','''sont''','''sous''','''souvent''','''soyez''','''soyons''','''specifique''','''specifiques''','''speculatif''','''stop''','''strictement''','''subtiles''','''suffisant''','''suffisante''','''suffit''','''suis''','''suit''','''suivant''','''suivante''','''suivantes''','''suivants''','''suivre''','''sujet''','''superpose''','''sur''','''surtout''','''t''','''ta''','''tac''','''tandis''','''tant''','''tardive''','''te''','''tel''','''telle''','''tellement''','''telles''','''tels''','''tenant''','''tend''','''tenir''','''tente''','''tes''','''tic''','''tien''','''tienne''','''tiennes''','''tiens''','''toc''','''toi''','''toi-même''','''ton''','''touchant''','''toujours''','''tous''','''tout''','''toute''','''toutefois''','''toutes''','''treize''','''trente''','''tres''','''trois''','''troisième''','''troisièmement''','''trop''','''très''','''tsoin''','''tsouin''','''tu''','''té''','''u''','''un''','''une''','''unes''','''uniformement''','''unique''','''uniques''','''uns''','''v''','''va''','''vais''','''valeur''','''vas''','''vers''','''via''','''vif''','''vifs''','''vingt''','''vivat''','''vive''','''vives''','''vlan''','''voici''','''voie''','''voient''','''voilà''','''voire''','''vont''','''vos''','''votre''','''vous''','''vous-mêmes''','''vu''','''vé''','''vôtre''','''vôtres''','''w''','''x''','''y''','''z''','''zut''','''à''','''â''','''ça''','''ès''','''étaient''','''étais''','''était''','''étant''','''état''','''étiez''','''étions''','''été''','''étée''','''étées''','''étés''','''êtes''','''être''','''ô''']
stop_words.append(',')
stop_words.append(')')
stop_words.append('(')
stop_words.append('.')
stop_words.append("'")
stop_words.append(":")

for w in words:
    if w in stop_words:
        sel = words == w.lower()
        sel = sel == False
        words = words[sel]

gaia.columns = ['lien_url','note_fb','text_gaia']

from threading import Thread
class nlp_text_gaia(Thread):
    def __init__(self,nb_file,gaia,words,nb_feat):
        Thread.__init__(self)
        self.nb_file = nb_file
        self.gaia = gaia
        self.words = words
        self.nb_feat = nb_feat
        self.fin = False

    def run(self):
        k = 1
        outecomes = pandas.DataFrame()
        outecomes['note_fb'] = numpy.array(self.gaia['note_fb'])
        for w in self.words:
            if k > (self.nb_file - 1) * self.nb_feat and k <= self.nb_file * self.nb_feat:
                sel = numpy.array(self.gaia['text_gaia'].apply(lambda x: w.lower() in x.lower()))
                outecomes[w] = sel.astype(int)
            k += 1
        outecomes.to_csv('C:/fake_news/doc/bdd_texte_' + str(self.nb_file) + '.csv',index = False,sep = ';')
        outecomes = pandas.DataFrame()
        outecomes['note_fb'] = numpy.array(self.gaia['note_fb'])
        self.fin = True


nb_total_file = int(len(words) / nb_feat) + 1
l_ntg = []
traitement = numpy.array(range(1,nb_total_file + 1))
sel = traitement % nb_worker == worker_id
traitement = traitement[sel]
for i in traitement:
    ntg = nlp_text_gaia(nb_file = i,gaia = gaia,words = words,nb_feat = nb_feat)
    ntg.run()
    # l_ntg.append(ntg)
    # while len(l_ntg) == nb_worker:
    #     for l in l_ntg:
    #         if l.fin:
    #             l_ntg.remove(l)