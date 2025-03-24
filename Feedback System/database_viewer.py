import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import io
from PIL import Image
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_resource
def database_connection(n):
    print("Fetching data from database ..!")
    connection_string = "mongodb+srv://shab:varis@cluster0.ivkgp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(connection_string,server_api=ServerApi('1'))
    db = client['medicinal_plant']
    collection = db['history']
    documents = list(collection.find().sort('time',-1))
    return documents, collection

st.set_page_config(layout="wide")


if 'filter' not in st.session_state:
    st.session_state.filter ="all"
    st.session_state.delete_list=[]
    st.session_state.update_list=[]

documents, collection=database_connection(1)

def feedback_buffer(id):
    new_value = st.session_state[id].split('[')[-1].rstrip(']')
    st.session_state.update_list.append((id,new_value))

def delete_buffer(id):
    st.session_state.delete_list.append(id)
    print(st.session_state.delete_list)

def apply_changes(mode):
    if mode:
        collection.delete_many({'_id': {'$in': st.session_state.delete_list}})
        for id,value in st.session_state.update_list:
            updated_doc = {'$set': {'feedback': value}}  
            collection.update_one({'_id': id}, updated_doc)
        database_connection.clear()
    st.session_state.delete_list.clear()
    st.session_state.update_list.clear()
    
tab1, tab2 = st.tabs(["Home", "Analysis"])

with tab1:
	head,save,delete,_,all,new=st.columns([5,1,1,0.3,0.5,0.5])

	with head:
		st.header("Prediction Logging",anchor=False)
	with save:
		if len(st.session_state.delete_list)>0 or len(st.session_state.update_list)>0:
			apply_changer=st.button("Save changes",use_container_width=True,type='primary',on_click=apply_changes,args=(1,))
	with delete:
		if len(st.session_state.delete_list)>0 or len(st.session_state.update_list)>0:
			delete_changer=st.button("Delete changes",use_container_width=True,type='secondary',on_click=apply_changes,args=(0,))
	with all:
		if st.button("All",use_container_width=True):
			st.session_state.filter="all"
	with new:
		if st.button("New",use_container_width=True):
			st.session_state.filter="new" 
	
	nmbr=0        
	for nbr,doc in enumerate(documents):
		data=list(doc.values())
		if st.session_state.filter=='all' or  len(data)==5:
			col1, col2, col3, col4 ,col5,col6= st.columns([0.4,1, 1, 1.8, 0.7,1],gap="small")  
		with col1:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('No',anchor=False)
			with st.container(border=True,height=200,):
				st.text(nmbr+1)  

		with col2:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('File Name',anchor=False)
			with st.container(border=True,height=200,):
				st.write(data[1])
		with col3:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('Prediction',anchor=False)
			with st.container(border=True,height=200,):
				st.write(data[3],)
			
		with col4:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('Image',anchor=False)
			with st.container(border=True,height=200,):
				image=Image.open(io.BytesIO(data[2]))
				st.image(image,use_column_width='never')
			
		with col5:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('Date',anchor=False)
			with st.container(border=True,height=200,):
				dt_object=datetime.fromisoformat(str(data[4]))
				st.write(dt_object.strftime("%-d-%B-%Y \n  %H:%M:%S"))
		with col6:
			with st.container(border=True):
				if nmbr==0:
					st.subheader('Feedback',anchor=False)
			with st.container(border=True,height=200):
				if len(data)==6 :
					index=0 if data[5]=='Correct' else  1
				else:
					index=None
				st.radio("prediction",[":green[Correct]",":red[Wrong]"],index=index,key=data[0]
				,on_change=feedback_buffer,args=(data[0],),label_visibility="hidden")
				st.write(' ')
				if data[0] not in st.session_state.delete_list:
					st.button("Delete",key=nmbr,type='primary',on_click=delete_buffer,args=(data[0],))
				else:
					st.button("Deleted",key=nmbr)
					
		nmbr+=1    

with tab2:
	documents = collection.find({"feedback": {"$exists": True}},{"feedback":1,"prediction":1,"_id":0})
	df=pd.DataFrame(columns=["Class","Prediction"])
	for i in documents:
		pos=len(df)
		df.loc[pos]=i.values()		
	df=df.groupby("Class")["Prediction"].value_counts().unstack(fill_value=0)
	colms=df.columns
	df["Total Prediction"]=df[colms[0]]+df[colms[1]] if len(colms) >1 else df[colms[0]] 
	df["Precision (%)"]=df["Correct"]/df["Total Prediction"] if 'Correct' in colms else 0
	_=df.rename(columns={"Correct":"Correct Prediction (TP)"},inplace=True) if 'Correct' in colms else df.rename(columns={"Wrong":"Wrong Prediction (FP)"},inplace=True)
	df=df[df.columns[[2,0,1,3]]] if len(colms)>1 else df[df.columns[[1,0,2]]]

	total=df["Total Prediction"].sum() 
	crct=df["Correct Prediction (TP)"].sum() if 'Correct' in colms else 0
	stat,grph=st.columns([0.5,0.5])
	with stat:
		st.header("",anchor=False)
		st.header(f"Total Accuracy : {round((crct/total)*100,2)}%",anchor=False)
		st.subheader(f"Total Predictions :{total}",anchor=False)
		st.subheader(f"True Positive : {crct}",anchor=False)
		st.subheader(f"False Positive : {total-crct}",anchor=False)
		st.subheader(f"Total Predicted Classes : {len(df)}",anchor=False)
	
	with grph:
		plt.style.use('fivethirtyeight') 
		fig, ax = plt.subplots()
		fig.patch.set_alpha(0) 
		# ax.set_facecolor((1, 1, 1, 0)) 
		fig.patch.set_facecolor('grey')  
		ax.pie([crct,total-crct],  labels=["Correct","Wrong"],
		autopct='%1.1f%%', startangle=90,explode=[0.01,0.01],textprops={'color': 'white'})
		# ax.axis('equal')  
		st.title("Accuracy",anchor=False)
		st.pyplot(fig)
	
	st.dataframe(df)