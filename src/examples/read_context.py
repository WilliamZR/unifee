from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage,get_wikipage_by_id
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--db_file', type=str)
	args = parser.parse_args()
	db =  FeverousDB(args.db_file)

	page_json = db.get_doc_json("Anarchism")
	#print(page_json)

	wiki_page = WikiPage("Anarchism", page_json)

	
	context_sentence_55 = wiki_page.get_context('table_0') # Returns list of Wiki elements
	print(context_sentence_55)

	context_text = [str(element) for element in context_sentence_55] # Get context text for each context element
	print(context_text)
	context_ids = [element.get_id() for element in context_sentence_55] # Get context id for each context element
	print(context_ids)

	prev_elements = wiki_page.get_previous_k_elements('sentence_5', k=4) #Gets Wiki element before sentence_5
	next_elements = wiki_page.get_next_k_elements('sentence_5', k=4) #Gets Wiki element after sentence_5
