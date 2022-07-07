
openFile()
function openFile()
{
	
	filenamefile_dir="/home/sujiwosa/Downloads/LNDS_June2022/LIDC-IDRI-Segmentation-master/lung-preprocessing/datapiv/Meta/nonnodulemhd.csv";
//	dirname_dir="/media/sujiwosa/3678DFE778DFA441/PIV_journal_revision/dirname.txt";
	filestring=File.openAsString(filenamefile_dir);
	rows=split(filestring, "\n");
	//dirstring=File.openAsString(dirname_dir);
	//dirrows=split(dirstring, "\n");
	//mhdfiles_dir="/media/sujiwosa/3678DFE778DFA441/PIV_journal_revision/original_full/";
	mhdfiles_dir="/home/sujiwosa/Downloads/LNDS_June2022/LIDC-IDRI-Segmentation-master/lung-preprocessing/datapiv/Clean/Image/";
	outPIVfiles_dir="/home/sujiwosa/Downloads/LNDS_June2022/LIDC-IDRI-Segmentation-master/lung-preprocessing/data_pivmhd_4x4/Clean/Image";
	print("---------------------");
	//for(j=0; j<dirrows.length; j++){
	//	outPIVfiles_dir="/media/sujiwosa/3678DFE778DFA441/PIV_journal_revision/pivdata/"+dirrows[j]+"/";
	//	print(outPIVfiles_dir);
	for(i=0; i<rows.length; i++){
		print(rows[i]);
		inputfilepath=mhdfiles_dir+rows[i]+".mhd";
		filewext=rows[i]+".mhd";
		print(inputfilepath);
		outputfilepath=outPIVfiles_dir+"/"+filewext;
		print(outputfilepath);		
		run("MHD/MHA...", "open="+inputfilepath);
		run("8-bit");
		run("PIV analysis", "window=4x4 diplay masking=0.50");
		selectWindow("Peak height");
		run("MHD/MHA ...", "save=["+outputfilepath+"]");		
		selectWindow("Peak height");
		close();
		selectWindow("Color coded orientation");
		close();
		selectWindow("Flow direction");	
		close();
		selectWindow("U");
		close();
		selectWindow("V");
		close();			
		opfilename=filewext;
		opfilename = replace(opfilename, ".mhd", ".raw"); 
		dirfile=split(opfilename, "/");
		print(dirfile[1]);
		selectWindow(dirfile[1]);
		close();				
	}
		print("---------------------");

	//	}
}



function processInputFile(inputFoFname){
	print("Hello Godfrey")
	FileReader fnr=new FileReader(inputFoFname)
	try (BufferedReader br = new BufferedReader(fnr)) {
    String line;
    while ((line = br.readLine()) != null) {
       System.out.println(line);
    }
}
}

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	// Do the processing here by adding your own code.
	// Leave the print statements until things work, then remove them.
	print("Processing: " + input + File.separator + file);
	print("Saving to: " + output);
}
