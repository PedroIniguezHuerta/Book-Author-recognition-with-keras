import os
import numpy as np
import re

#####################################################################
# Class containing all algorithms to generate a dataset from a list 
# of PDFs files of different authors.
#####################################################################
class bookPreprocessor:
    def __init__(self, directory):
        self.cdirectory = directory
        self.singleList = []
        self.subdirs = {}
        self._updatePaths()
        self.dataset = "authors.csv"        
        self.traindataset = "trainplus.csv"
        self.testdataset = "testplus.csv"       

    #####################################################################
    # update the working directories names
    #####################################################################
    def _updatePaths(self):
        self.PDF_DIR = self.cdirectory
        self.TXT_DIR = self.cdirectory + "/txtfiles/"
        self.CLEANTXT_DIR = self.cdirectory + "/cleantxtfiles/"
    

    #####################################################################
    # read a file from disk and return a list of lines
    #####################################################################
    def _readFile(self, filename):
        fr = open(filename,"r")
        text = fr.read()
        fr.close()
        return text

    #####################################################################
    # Remove special characters, blanks lines, etc.
    #####################################################################
    def _cleanFile(self, text, header, footer):
        cleanText = ""

        text = text.replace(header,"")
        text = text.replace(footer,"")

        lines = text.split("\n")

        ignoreCtl = False
        for line in lines:
            #print(line)
            try:
                nline = line.replace(" ","").strip()

                if len(line) == 0 or int(nline) > 0:
                    ignoreCtl = True
            except:
                if line[0] == '\x0c':
                    line = line[1:]

                if len(line) > 0:
                    ignoreCtl = False
                    cleanText += line + " "
   
        phrases = cleanText.split(".")
        finalText = ""

        if header != "":
            finalText = header + ".\n"

        for phrase in phrases:
            phrase = phrase.strip()
            if len(phrase) > 0:
                finalText += phrase +  ".\n"
                #print(phrase+".")

        if footer != "":
            finalText += footer + ".\n"
        return finalText


    #####################################################################
    # write a list of text lines into a file
    #####################################################################
    def _writeFile(self, filename, clean_text):
        fw = open(filename,"w")
        fw.write(clean_text)
        fw.close()


    #####################################################################
    # Directory listing function
    #####################################################################
    def _listFiles(self, directory):
        return os.listdir(directory)

    #####################################################################
    # Convert all PDFs files into text format
    #####################################################################
    def _pdftotext(self):
        files = self._listFiles(self.PDF_DIR)
        for cfile in files:
            if cfile.endswith(".pdf"):
                print("converting file:", cfile)
                output_txt_file = self.TXT_DIR + cfile[:-4] + ".txt"

                os.system("pdftotext " + cfile + " " + output_txt_file)

                #####################################################################
                # read the converted txt file and remove header, tail, line feeds, blank lines, etc.
                #####################################################################
                print("cleaning up file:", output_txt_file)
                # TODO: Automatically extract header and footer to be removed as the most repetitive lines
                header = ""
                footer = ""
                text = self._readFile(output_txt_file)
                clean_text = self._cleanFile(text,header, footer)

                #####################################################################
                # Store cleaned file in subdirectories and save them as the class names
                #####################################################################
                index = cfile.find("-")
                subdir = ""

                if index > 0:
                   subdir = cfile[:index]
                   self.subdirs[subdir] = subdir   # save class name in dictionary
                   subdir += "/"

                if os. path. isdir(self.CLEANTXT_DIR + subdir) == False:
                    os.system("mkdir " + self.CLEANTXT_DIR + subdir)

                output_clean_file = self.CLEANTXT_DIR + subdir + cfile[:-4] + "-clean.txt"
                print("generating file:", output_clean_file)

                self._writeFile(output_clean_file, clean_text)

    #####################################################################
    # convert the text files into CSV format and label each row with the proper class/author
    #####################################################################
    def _convertFilesToCSV(self, directory, subdir, label):
        files = self._listFiles(directory+"/"+subdir)
        lines = 0

        for cfile in files:
            f = open(directory+"/"+subdir+"/"+cfile,"r")
            #fclean = f.read().replace(";",",").replace("\n","").replace("\r","")
            fclean = f.read().replace(";",",")
            phrases = fclean.split("\n")

            for phrase in phrases:
                clean_text = re.sub("^([0-9A-Za-z\u00C0-\u017F\ ,.\;'\-()\s\:\!\?\"])+", ' ', phrase)
                clean_text = clean_text.strip()
                if len(clean_text) > 2:
                    self.singleList.append(clean_text[:-1] + ";" + label);
                    lines += 1

            f.close()
        return lines

    #####################################################################
    # Randomly mix phrases of dataset to allow machine learning algorithms works better
    #####################################################################
    def _mixPhrases(self):
        iterations = len(self.singleList)
        mixedFile = "phrase;label\n"

        for i in range(0,iterations):
            pos = np.random.randint(iterations-i)
            mixedFile += self.singleList[pos] + "\n"
            del self.singleList[pos]

        print("mixed file size=",len(mixedFile))
        return mixedFile
    

    #####################################################################
    # Generate a single dataset from the list of PDFs files
    #####################################################################
    def _generateDataset(self, directory, filename):
        print("Generating dataset:",filename)
        print("Classes:",self.subdirs)
        class_counter = 0
        total_lines = 0
        self.csvfile = ""

        for subdir in self.subdirs:
            # Update self.singleList with the lines of current subdir
            lines = self._convertFilesToCSV(directory, subdir, str(class_counter))
            total_lines +=  lines
            class_counter += 1

            print(subdir + " lines=",lines)

        print("total_lines =", total_lines)
        mixedFile = self._mixPhrases()

        fw = open(filename, "w")
        fw.write(mixedFile)
        fw.close()

    #####################################################################
    # Split dataset into train and test datasets
    #####################################################################
    def _splitDataset(self, inputDataset, trainDataset, testDataset, percentage):
        r = open(inputDataset,"r")
        text = r.read()
        r.close()

        lines = text.split("\n")

        train_text = lines[0] + "\n"
        test_text = lines[0] + "\n"

        print("train_text",train_text)
        print("test_text",test_text)

        total_lines = len(lines)
        train_lines = int(total_lines*percentage)
        
        print("total lines:",total_lines)
        print("train lines:",train_lines)

        for i in range(1,total_lines):
            if i < train_lines:
                train_text += lines[i] + "\n"
            else:
                test_text += lines[i] + "\n"

        w = open(trainDataset,"w")
        w.write(train_text)
        w.close()

        w = open(testDataset,"w")
        w.write(test_text)
        w.close()

    #####################################################################
    # Create main output/working subdirectories
    #####################################################################
    def _createWorkingDirectories(self):
        os.system("mkdir " + self.TXT_DIR)
        os.system("mkdir " + self.CLEANTXT_DIR)

    #####################################################################
    # external method to generate the dataset
    #####################################################################
    def preprocessBooks(self):
        self._createWorkingDirectories()
        self._pdftotext()
        self._generateDataset(self.CLEANTXT_DIR, self.dataset)
        self._splitDataset(self.dataset, self.traindataset, self.testdataset, 0.8)


#####################################################################
# main
#####################################################################
bp = bookPreprocessor(".")
bp.preprocessBooks()
