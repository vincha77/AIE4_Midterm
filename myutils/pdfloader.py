"""
pdfloader.py
This class loads a list of pdf documents passed in 
and returns a list of parsed text for these docs

User can provide one of a few options to load pdf...
pypdf or pymupdf

"""

# importing required classes
import os
from typing import List

from pypdf import PdfReader 
import pymupdf


VALID_PDF_MODULES = ['pypdf', 'pymupdf']


class TextFromPdf:
    '''
    this class converts a list of pdf documents into a list of text documents
    '''
    def __init__(self,
                 pdfmodule: str,
                 list_of_pdf_docs: List[str]):

        # validate pdfmodule
        if pdfmodule in VALID_PDF_MODULES:
            self.pdfmodule = pdfmodule
        else:
            print(f'ERROR: pdfmodule must be one of {VALID_PDF_MODULES}')
            raise Exception
    
        # validate input list
        if isinstance(list_of_pdf_docs, list) and len(list_of_pdf_docs) > 0:
            self.list_of_pdf_docs = list_of_pdf_docs
        else:
            print('ERROR: expecting a non-empty list of pdf names to be passed in')
            raise Exception
        return
    
    def process_single_pdf_with_pypdf(self, pdfdoc):
        # check if file exists; if not return None
        if os.path.isfile(pdfdoc):
            pass
        else:
            print(f'Warning: pdf file {pdfdoc} does not exist...skipping to next pdf file')
            return None
        reader = PdfReader(pdfdoc)
        numpages = len(reader.pages)
        thistext = ''
        for pagecount in range(0, numpages):
            page = reader.pages[pagecount]
            pagetext = page.extract_text()
            thistext = thistext + '\n ' + pagetext  # adding a line break
            # print('\n')
            # print(thistext)
        return thistext
    
    
    def process_single_pdf_with_pymupdf(self, pdfdoc):
        # check if file exists; if not return None
        if os.path.isfile(pdfdoc):
            pass
        else:
            print(f'Warning: pdf file {pdfdoc} does not exist...skipping to next pdf file')
            return None

        doc = pymupdf.open(pdfdoc) # open a document
        thistext = ''
        for page in doc:
            pagetext = page.get_text() # get plain text (is in UTF-8)
            thistext = thistext + '\n ' + pagetext  # adding a line break
            # print('\n')
            # print(thistext)
        return thistext
    
    def process_all_pdfs(self):
        list_of_texts = []
        for pdfdoc in self.list_of_pdf_docs:
            pdftext = self.process_single_pdf(pdfdoc)
            if pdftext is not None:
                list_of_texts.append([pdftext])
        return list_of_texts
