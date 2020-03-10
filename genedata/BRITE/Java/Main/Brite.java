/****************************************************************************/
/*                  Copyright 2001, Trustees of Boston University.          */
/*                               All Rights Reserved.                       */
/*                                                                          */
/* Permission to use, copy, or modify this software and its documentation   */
/* for educational and research purposes only and without fee is hereby     */
/* granted, provided that this copyright notice appear on all copies and    */
/* supporting documentation.  For any other uses of this software, in       */
/* original or modified form, including but not limited to distribution in  */
/* whole or in part, specific prior permission must be obtained from Boston */
/* University.  These programs shall not be used, rewritten, or adapted as  */
/* the basis of a commercial software or hardware product without first     */
/* obtaining appropriate licenses from Boston University.  Boston University*/
/* and the author(s) make no representations about the suitability of this  */
/* software for any purpose.  It is provided "as is" without express or     */
/* implied warranty.                                                        */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
/*  Author:     Alberto Medina                                              */
/*              Anukool Lakhina                                             */
/*  Title:     BRITE: Boston university Representative Topology gEnerator   */
/*  Revision:  2.0         4/02/2001                                        */
/****************************************************************************/

package Main;

import Model.*;
import Graph.*;
import Export.*;
import Topology.*;
import Util.*;
import Import.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.HashMap;

final public class Brite { 
    
    public static void main(String args[]) {
      String filename="";
      String outFile ="";
      String seedFile="";
      
      /*get config file, get output file*/
      try { 
	  filename = args[0];
	  outFile = args[1];
	  seedFile = args[2];
      }
      catch (Exception e) { 
	  Util.ERR("Usage:  java Main.Main config_file output_file seed_file");
	  System.exit(0);
      }
            
      RandomGenManager rgm = new RandomGenManager();
      rgm.parse(seedFile);

      	      
      /*create our glorious model and give it a random gen manager*/
      Model m = ParseConfFile.Parse(filename);
      m.setRandomGenManager(rgm);
      
      /*now create our wonderful topology. ie call model.generate()*/
      Topology t = new Topology(m);
      
      /*check if our wonderful topology is connected*/
      Util.MSGN("Checking for connectivity:");
      Graph g = t.getGraph();
      boolean isConnected = (g.isConnected());
      if (isConnected)
	System.out.println("\tConnected");
      else System.out.println("\t***NOT*** Connected");
      
      
      /*export to brite format outfile*/
      HashSet exportFormats = ParseConfFile.ParseExportFormats();
      ParseConfFile.close();
      if (exportFormats.contains("BRITE")) {
	  Util.MSG("Exporting Topology in BRITE format to: " + outFile+".brite");
	  BriteExport be = new BriteExport(t, new File(outFile+".brite"));
	  be.export();
      }
      /*export to otter format outfile*/
      if (exportFormats.contains("OTTER")) {
	  Util.MSG("Exporting Topology in Otter Format to: " + outFile+".odf");
	  OtterExport oe = new OtterExport(t, new File(outFile+".odf"));
	  oe.export();
      }
      
      /*outputting seed file*/
      Util.MSG("Exporting random number seeds to seedfile");
      rgm.export("last_seed_file", seedFile);
	
      
      /*we're done (and hopefully successfully)*/
      Util.MSG("Topology Generation Complete.");

      //t.dumpToOutput();

  }
  
  
}





