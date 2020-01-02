# Inspired by https://fr.slideshare.net/jimarlow
import json
from IPython.display import display, Markdown, HTML


def MakeLink(inFName,volume):
	dirName = "../Notebooks_"+volume+"/"; extension = ".ipynb"
	print("Notebook ["+inFName+"]("+dirName+inFName+extension+") "
		+ ", from volume "+ volume + " [Summary]("+dirName+"Summary"+extension+") " )

def Info(volume):
	if volume in ['NonDiv','Div','Algo','Repro']:
		return """
**Acknowledgement.** The experiments presented in these notebooks are part of ongoing research, 
some of it with PhD student Guillaume Bonnet, in co-direction with Frederic Bonnans.

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""
	elif volume=='FMM':
		return """
This Python&reg; notebook is intended as documentation and testing for the [HamiltonFastMarching (HFM) library](https://github.com/mirebeau/HamiltonFastMarching), which also has interfaces to the Matlab&reg; and Mathematica&reg; languages. 
More information on the HFM library in the manuscript:
* Jean-Marie Mirebeau, Jorg Portegies, "Hamiltonian Fast Marching: A numerical solver for anisotropic and non-holonomic eikonal PDEs", 2019 [(link)](https://hal.archives-ouvertes.fr/hal-01778322)

Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
"""

VolumeTitle = {
'FMM':"Fast Marching Methods",
'NonDiv':"Non-Divergence form PDEs",
'Div':"Divergence form PDEs",
'Algo':"Algorithmic tools",
'Repro':"Reproducible research"
}


VolumeFilenames = {
'FMM':[
    "Isotropic","Riemannian","Rander","AsymmetricQuadratic",
    "Curvature","Curvature3","DeviationHorizontality",
    "HighAccuracy","SmartIO","Sensitivity","SensitivitySL",
    "Illusion","Tubular","FisherRao","DubinsZermelo",
    "Seismic",
],
'NonDiv':[
	"MonotoneSchemes1D","LinearMonotoneSchemes2D","NonlinearMonotoneFirst2D","NonlinearMonotoneSecond2D",
	"MongeAmpere","OTBoundary1D","EikonalEulerian"
],
'Div':["Elliptic","EllipticAsymmetric","VaradhanGeodesics"],
'Algo':["TensorSelling","TensorVoronoi",
"SternBrocot","VoronoiVectors",
"Dense","Sparse","Reverse","ADBugs",
"SubsetRd"],
'Repro':[],
}

RepositoryDescription = """
**Github repository** to run and modify the examples on your computer.
[AdaptiveGridDiscretizations](https://github.com/Mirebeau/AdaptiveGridDiscretizations)\n
"""

def displayTOC(inFName,volume):
	with open(inFName+".ipynb", encoding='utf8') as data_file:
		data = json.load(data_file)
	contents = []
	for c in data['cells']:
		s=c['source']
		if len(s)==0:
			continue
		line1 = s[0].strip()
		if line1.startswith('#'):
			count = line1.count('#')-1
			plainText = line1[count+1:].strip()
			if plainText[0].isdigit() and int(plainText[0])!=0:
				link = plainText.replace(' ','-')
				listItem = "  "*count + "* [" + plainText + "](#" + link + ")"
				contents.append(listItem)

	contents = ["[**Summary**](Summary.ipynb) of volume " + VolumeTitle[volume] + ", this series of notebooks.\n",
	"""[**Main summary**](../Summary.ipynb) of the Adaptive Grid Discretizations 
	book of notebooks, including the other volumes.\n""",
	"# Table of contents"] + contents + ["\n\n"+Info(volume)]

	return "\n".join(contents)
	

# 	display(Markdown("[**Summary**](Summary.ipynb) of this series of notebooks. "))
# 	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """))
# #	display(HTML("<a id = 'table_of_contents'></a>"))
# 	display(Markdown("\n# Table of contents"))
# 	display(Markdown("\n".join(contents)))
# 	display(Markdown("\n\n"+Info(volume)))

def displayTOCs(volume,subdir=""):
	inFNames = VolumeFilenames[volume]
	contents = []
	part = ""
	part_counter = 0
	part_numerals = "ABCDEFGHIJK"
	chapter_counter = 0
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for _inFName in inFNames:
		inFName = _inFName+".ipynb"
		with open(subdir+inFName, encoding='utf8') as data_file:
			data = json.load(data_file)
			# Display the chapter
			s=data['cells'][0]['source']
			sec = s[2][len("## Part : "):]
			if sec!=part:
				part=sec
				contents.append("### "+part_numerals[part_counter]+". "+part)
				part_counter+=1
				chapter_counter=0
			else:
				chapter_counter+=1
			chapter = s[3][len("## Chapter : "):].strip()
			contents.append(" " + "* "+chapter_numerals[chapter_counter] +
				". [" + chapter + "](" + inFName + ")")
			# Display the sub chapters
			for c in data['cells']:
				s = c['source']
				if len(s)==0: 
					continue
				line1 = s[0].strip()
				if line1.startswith('##') and line1[3].isdigit() and int(line1[3])!=0:
					contents.append(" "*2 + line1[len("## "):])
			contents.append("\n")

	contents = [RepositoryDescription,"# Table of contents",
		"[**Main summary**](../Summary.ipynb), including the other volumes of this work. "]+contents;
	return "\n".join(contents)
#	display(Markdown(RepositoryDescription))
#	display(Markdown("# Table of contents"))
#	display(Markdown("""[**Main summary**](../Summary.ipynb), including the other volumes of this work. """ ))
#	display(Markdown("\n".join(contents)))

def displayTOCss():
	extension = '.ipynb'
	contents = []
	volume_numerals = "1234"
	part_numerals = "ABCDEFGHIJK"
	chapter_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]
	for volume_counter,volume in enumerate(['FMM','NonDiv','Div','Algo']):
		dirName = 'Notebooks_'+volume+'/'
		part = ""
		part_counter = 0

		inFName = dirName+'Summary'+extension
		with open(inFName, encoding='utf8') as data_file:
			data = json.load(data_file)
			s = data['cells'][0]['source']
			volumeTitle = s[2][len("# Volume : "):]
			contents.append("### " + volume_numerals[volume_counter]+". "+
				"["+volumeTitle+"]("+inFName+")")

		# Display parts and chapters
		for _inFName in VolumeFilenames[volume]:
			inFName = dirName+_inFName+extension
			with open(inFName, encoding='utf8') as data_file:
				data = json.load(data_file)
				# Display the chapter
				s=data['cells'][0]['source']
				sec = s[2][len("## Part : "):]
				if sec!=part:
					part=sec
					contents.append(" * "+part_numerals[part_counter]+". "+part)
					part_counter+=1
					chapter_counter=0
				else:
					chapter_counter+=1
				chapter = s[3][len("## Chapter : "):].strip()
				contents.append("  " + "* "+chapter_numerals[chapter_counter] +
					". [" + chapter + "](" + inFName + ")")
		contents.append("")

	contents = ["# Table of contents"]+contents
	return "\n".join(contents)
#	display(Markdown("# Table of contents"))
#	display(Markdown("\n".join(contents)))







