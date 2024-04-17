finalScores = {}
with open('output.txt','r')as f:
        counter =0
        for line in f:
            lineList = line.split()
            songNameList = lineList[: len(lineList) - 2] #remove last 2 elements of list which are genre and score --> leaves u with just song name as a list
            if counter ==0:#handle first line
                #first line will always be the one chosen bc we have sorted by descending similarity score
                songName = ''
                for i in songNameList:
                    songName+=i
                    songName+=' '
                finalScores[songName] = lineList[-2]#second to last element is genre
            counter+=1
            if counter ==5: #CHANGE THIS BY HOW MANY GENRES THE DATASET WILL HAVE !!!
                songName = ''
                for i in songNameList:
                    songName+=i
                    songName+=' '
                finalScores[songName] = lineList[-2]#second to last element is genre
                counter=1
print(finalScores)
with open('final_output.txt','w')as f:
        for song in finalScores:
            f.write(f"{song} {finalScores[song]}\n")