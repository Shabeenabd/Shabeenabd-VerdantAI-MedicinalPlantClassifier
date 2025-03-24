from downloader import download
classes=['Aanachuvadi', 'Aeriku', 'African coriander', 'Aloevera', 'Amukram', 'Anadaman thipali', 'Arayal', 'Arogyapacha', 'Athi', 'Ayyappana', 'Brahmi', 'Chakarakoli', 'Chandanam', 'Changalamparanda', 'Chathuramulla', 'Chembarati', 'Cherukadaladi', 'Cherula', 'Cherupulladi', 'Cherupunna', 'Cheruthekku', 'Chittalodakam', 'Chittamruthu', 'Communist_pacha', 'Curry leaves', 'Grambu', 'Guava', 'Idampiri valampiri', 'Ilakalli', 'Ilamulli', 'Illipa', 'Insulin plant', 'Iruveli', 'Ithi', 'Javathippali', 'Kaara', 'Kadaladi', 'Kaiyonni', 'Karikudangal', 'Karimaram', 'Karinjotta', 'Kasthurivenda', 'Kodithoova', 'Koovalam', 'Krishnatulasi', 'Kudangal', 'Kumbil', 'Kurumulaku', 'Kurunthotti', 'Mandaram', 'Mango Tree', 'Maramanjal', 'Maroti', 'Moovila', 'Mukooti', 'Munja', 'Murikoodi', 'Muyalchevian', 'Naruneendi', 'Neelakoduveli', 'Neem', 'Neermaruthu', 'Nelli', 'Nilapana', 'Njota njodiyan', 'Padathali', 'Palakappayani', 'Panikoorka', 'Pathimugham', 'Peraal', 'Perumkurumba', 'Plasu', 'Poochameesatulasi', 'Pulimaram', 'Puliyarila', 'Puthranjeeva', 'Rakthachandanam', 'Ramacham', 'Sarvasugandi', 'Shangukuppi', 'Shankupushpam', 'Shathavari', 'Thanni', 'Thetti', 'Thipali', 'Thottavadi', 'Ummam', 'Uzhinja', 'Vathamkolli', 'Vayamkatha', 'Vellakoduveli', 'Vembada', 'Vennochi', 'Vettupala', 'arootha', 'chemaram', 'gugglu', 'kadakampala', 'peruvanam', 'ponnamkannicheera', 'pushkaramulla', 'ushamalari', 'yesanku']

plant=input("Enter the Plant name to download : ")
nmbr=input("Enter the number of images to download : " )
image_number=int(nmbr) if nmbr else 5 

print("PRESS : \ns - skip image\nd - download image\nq - for query\nn - next class\nx - exit")

query=plant if plant else classes

if __name__=="__main__":
	download(query=query,limit=image_number)
