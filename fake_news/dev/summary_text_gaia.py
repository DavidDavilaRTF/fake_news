import pandas
import numpy
import os

class summary:
    def __init__(self,folder_path,filename_path,nb_file):
        self.folder_path = folder_path
        self.filename_path = filename_path
        self.nb_file = nb_file
        self.outecomes = pandas.DataFrame()

    def merge(self):
        for i in range(1,self.nb_file + 1):
            temp = pandas.read_csv(self.folder_path + '/' + self.filename_path + '_' + str(i) + '.csv', sep = ';',engine = 'python',encoding='utf-8')
            note_fb = numpy.array(temp['note_fb'])
            temp = temp.drop('note_fb',axis = 1)
            self.outecomes = pandas.concat([self.outecomes,temp],axis = 1)
            print((i,self.nb_file))
        self.outecomes['note_fb'] = note_fb
        self.outecomes.to_csv('C:/fake_news/doc/bdd_text.csv',sep = ';',index = False)
    
    def delete(self):
        for i in range(1,self.nb_file + 1):
            os.remove(self.folder_path + '/' + self.filename_path + '_' + str(i) + '.csv')

s = summary('C:/fake_news/doc','bdd_texte',219)
s.merge()
s.delete()