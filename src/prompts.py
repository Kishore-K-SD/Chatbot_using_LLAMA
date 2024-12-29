'''
===========================================
        Module: Prompts collection
===========================================
'''
qa_template = """Based on the given context, provide an answer that includes all relevant information. Ensure that any timestamps associated with the statements fetched from the vector database are included *directly after the corresponding text*.

Context: {context}
Question: {question}

Format your answer as follows:

For each piece of information extracted from the context, append its corresponding timestamp(s) immediately after the text. If a piece of information spans multiple timestamps, list all applicable timestamps separated by a hyphen.

"""
