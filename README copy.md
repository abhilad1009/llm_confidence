LLM are non deterministic and when performing extraction, we cannot provide confidence measure on the extracted values.
In traditional discriminator based system, we find the localization and confidence percentage of what is being extracted e.g. traditional ocr models

We want to create an llm extractor which provides confidence measure in similar fashion. Here is how we want to achieve this:
1) Finding anchor: whatever we are extracting will need some context based on text around that value. We use which token are contributing i.e. provide anchor point from which the value is extracted.
2) Finding localization: Once the anchor is found, we can identify the extracted value in the anchor paragraph. And we replace the value to be extracted in anchor with random values multiple times and ideally it should extract those values everytime if anchor is correct. Thus we found the location. 
3) Once location confirms that anchor is correct, we change the anchor text around the value with random words. If the extraction is using correct reasoning and context, if the anchor is changed, the value will not be extracted. We repeat this multiple times and confirm anchor.
4) Once both are confirmed, we can do averaging of those runs to give a confidence measure of extraction.

Create a python project to this. We can use libraries like hugging face or if this requires bespoke implementation, then use pytorch.
