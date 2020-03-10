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
/*  Author:     Anukool Lakhina                                             */
/*              Alberto Medina                                              */
/*  Title:     BRITE: Boston university Representative Topology gEnerator   */
/*  Revision:  2.0         4/02/2001                                        */
/****************************************************************************/
package GUI;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.border.LineBorder;
import java.lang.Runtime;   //so that we can call BriteC++ or BriteJava executable in native format
import java.io.*;           //to redirect stdout for runtime process (c++ or java)



public final class Brite extends JDialog implements ActionListener, Runnable
{
    
    public void init()
    {
	getContentPane().setLayout(null);
	getContentPane().setBackground(new java.awt.Color(204,204,204));
	setSize(494,530);
	JLabel1.setText("Topology Type:");
	getContentPane().add(JLabel1);
	JLabel1.setForeground(java.awt.Color.black);
	JLabel1.setFont(new Font("SansSerif", Font.BOLD,   12));
	JLabel1.setBounds(36,12,156,22);
	getContentPane().add(TopologyType);
	TopologyType.setFont(new Font("SansSerif", Font.PLAIN, 12));
	TopologyType.setBounds(170,12,202,26);
	TopologyType.addActionListener(this);
	getContentPane().add(ePanel);
    
	/*BEGIN: run C++ or Java exe choice*/
	getContentPane().add(ExeChoicesComboBox);
	ExeChoicesComboBox.setFont(new Font("SansSerif", Font.PLAIN,   12));
	ExeChoicesComboBox.setBounds(220,474,110,21);
	/*END: C++ or Java exe choice*/
	
	/*BEGIN: Build Topology Button*/
	BuildTopology.setText("Build Topology");
	BuildTopology.setActionCommand("Build Topology");
	BuildTopology.setBorder(lineBorder1);
	getContentPane().add(BuildTopology);
	BuildTopology.setForeground(java.awt.Color.black);
	BuildTopology.setFont(new Font("SansSerif", Font.PLAIN, 12));
	BuildTopology.setBounds(348,474,108,21);
	BuildTopology.addActionListener(this);
	/*END: Build Topology Button*/
	
	getContentPane().add(logo);
	logo.setBorder(null);
	logo.setBounds(389, 2, 67, 65);
	logo.addActionListener(this);

	getContentPane().add(JTabbedPane1);
	JTabbedPane1.setBackground(new java.awt.Color(153,153,153));
	JTabbedPane1.setBounds(24,48,432,315);
	JTabbedPane1.add(asPanel);
	JTabbedPane1.add(rtPanel);
	JTabbedPane1.add(tdPanel);
	JTabbedPane1.add(buPanel);
	JTabbedPane1.setTitleAt(0,"AS");
	JTabbedPane1.setTitleAt(1,"Router");
	JTabbedPane1.setTitleAt(2,"Top Down");
	JTabbedPane1.setTitleAt(3, "Bottom Up");
	JTabbedPane1.setSelectedIndex(0);
	JTabbedPane1.setSelectedComponent(asPanel);
	rtPanel.EnableComponents(false);
	rtDisabled=true;
	hDisabled=true;
	tdPanel.EnableComponents(false);
	buPanel.EnableComponents(false);
      
	
	HelpButton.setText("Help");
	HelpButton.setBorder(lineBorder1);
	getContentPane().add(HelpButton);
	HelpButton.setForeground(java.awt.Color.black);
	HelpButton.setFont(new Font("SansSerif", Font.PLAIN, 12));
	HelpButton.setBounds(24, 474, 50,21);
	HelpButton.addActionListener(this);

	//getContentPane().add(statusDialog);
	//statusDialog.setSize(200, 200);
	//	statusDialog.setVisible(false);
	

	//create status window where output of executable will be written
	sd.setSize(400,200);
	sd.setVisible(false);
	
	aboutPanel.setSize(300,300);
	aboutPanel.setVisible(false);

	
	hPanel.setSize(500,500);
	hPanel.setVisible(false);
	setTitle("Boston University Representative Internet Topology Generator (BRITE)");
    }

    

    public void actionPerformed(ActionEvent e) { 
      if (e.getSource().equals(HelpButton)) {
	hPanel.setVisible(true);
	return;
      }
      if (e.getSource().equals(logo)) {
	  aboutPanel.setVisible(true);
	  return;
      }
      String level = (String)TopologyType.getSelectedItem();
	if (e.getSource().equals(TopologyType)) {
	    level = (String)TopologyType.getSelectedItem();
	    if (level.equals(AS_TOPOLOGY)){
		JTabbedPane1.setSelectedComponent(asPanel);
		rtPanel.EnableComponents(false);
		tdPanel.EnableComponents(false);
		buPanel.EnableComponents(false);
		asPanel.EnableComponents(true);
		
	    }
	    else if (level.equals(ROUTER_TOPOLOGY)){
		JTabbedPane1.setSelectedComponent(rtPanel);
		tdPanel.EnableComponents(false);
		buPanel.EnableComponents(false);
		asPanel.EnableComponents(false);
		rtPanel.EnableComponents(true);
		
	    }
	    else if (level.equals(TOPDOWN_TOPOLOGY)) {
		JTabbedPane1.setSelectedComponent(tdPanel);
		tdPanel.EnableComponents(true);
		buPanel.EnableComponents(false);
		asPanel.EnableComponents(true);
		asPanel.EnableBW(false);
		rtPanel.EnableComponents(true);
		rtPanel.EnableBW(false);
	    }
	    else if (level.equals(BOTTOMUP_TOPOLOGY)) {
		JTabbedPane1.setSelectedComponent(buPanel);
		buPanel.EnableComponents(true);
		asPanel.EnableComponents(false);
		tdPanel.EnableComponents(false);
		rtPanel.EnableComponents(true);
	
	    }
	}

	else if (e.getSource().equals(BuildTopology)) {
	    String file = ((String)ePanel.ExportLocation.getText()).trim();
	    
	    if (file.equals("") || file==null){
		JOptionPane.showMessageDialog(this, "Error:  Missing Export File", "Error", JOptionPane.ERROR_MESSAGE);
		return;
	    }
	    
	    if (!ePanel.isBriteFormat() && !ePanel.isOtterFormat()) {
		JOptionPane.showMessageDialog(this, "Error: Must specify atleast one output format", "Error", JOptionPane.ERROR_MESSAGE);
		return;
	    }
	     sd.getTextArea().setText("");
	     if (!sd.isVisible())
	       sd.setVisible(true);
	     sd.repaint();
	     
	    String args = " GUI_GEN.conf  "+file;
	    BuildTopology.setEnabled(false);
	   
	    MakeConfFile(level);
	    
	  
	      runThread = new Thread(GUI.Brite.this);
	      runThread.setPriority(Thread.MAX_PRIORITY);
	      runThread.start();
	  
	    //  runExecutable(args);
	    
	}
    }
    
    public void run() {
	String file = ((String)ePanel.ExportLocation.getText()).trim();
	String args = " GUI_GEN.conf " + file;
			    
	runExecutable(args);
	
    }
    
    private void MakeConfFile(String topologyType) {
	try {
	    BufferedWriter bw = new BufferedWriter(new FileWriter(new File("GUI_GEN.conf")));
	    bw.write("#This config file was generated by the GUI. ");
	    bw.newLine();  bw.newLine();
	    bw.write("BriteConfig");
	    bw.newLine(); bw.newLine();
	    if (topologyType.equals(TOPDOWN_TOPOLOGY)) {
		tdPanel.WriteConf(bw);
		bw.newLine();
		asPanel.WriteConf(bw);
		bw.newLine();
		rtPanel.WriteConf(bw);
	    }
	    else if (topologyType.equals(BOTTOMUP_TOPOLOGY)) {
		buPanel.WriteConf(bw);
		bw.newLine();
		rtPanel.WriteConf(bw);
		bw.newLine();
	    }
	    else if (topologyType.equals(AS_TOPOLOGY))
		asPanel.WriteConf(bw);
	    else if (topologyType.equals(ROUTER_TOPOLOGY))
		rtPanel.WriteConf(bw);
	
	    bw.newLine();
	    bw.write("BeginOutput");
	    bw.newLine();
	    bw.write("\tBRITE = ");
	    if (ePanel.isBriteFormat())
		bw.write("1 ");
	    else bw.write("0 " );
	    bw.write("\t #1=output in BRITE format, 0=do not output in BRITE format");
	    bw.newLine();
	    bw.write("\tOTTER = ");
	    if (ePanel.isOtterFormat())
		bw.write("1 " );
	    else bw.write("0 ");
	    bw.write("\t #1=Enable visualization in otter, 0=no visualization");
	    bw.newLine();
	    bw.write("EndOutput");
	    bw.newLine();
	    bw.close();
	}
	catch (IOException e) { 
	    System.out.println("[BRITE ERROR]:  Cannot create config file. " + e);
	    return;
	}
	    
    }
    

    private void runExecutable(String args) {
	
      
	String cmdExe ="java -Xmx256M -classpath $CLASSPATH:.:Java/ Main.Brite ";
	boolean runC = false;

	if ( ((String)ExeChoicesComboBox.getSelectedItem()).equals(CPPEXE)) {
	    runC= true;
	    cmdExe = "./C++/brite ";
	}
	System.out.println(cmdExe);
	
	String runThis = cmdExe+ args;
	Runtime r= Runtime.getRuntime();
	try {
	   
	    if (runC)
		runThis +=" C++/seed_file"; /*C++ version requires a seed file*/
	    else
		runThis+=" Java/seed_file"; /*java version also requires a java seed*/

	    System.out.println("[MESSAGE]: GUI starting executable: "+runThis); 
	    p = r.exec(runThis);
	    InputStream in = p.getInputStream();
	    BufferedReader brIn = new BufferedReader(new InputStreamReader(in));
	    String line;
	  
	    JTextArea sdLog = sd.getTextArea();
	    
	    while ((line= brIn.readLine())!=null) {
		sdLog.append(line+"\n");
		Rectangle rect = sdLog.getVisibleRect();
		
		int a = sdLog.getScrollableBlockIncrement(rect, SwingConstants.VERTICAL, 1);
		rect.setLocation((int)rect.getX(), (int)rect.getY()+a);
		sdLog.scrollRectToVisible(rect);
		System.out.println(line);
	    }
	    InputStream err = p.getErrorStream();
	    BufferedReader brErr = new BufferedReader(new InputStreamReader(err));
	    while ((line=brErr.readLine())!=null) {
		sdLog.append(line+"\n");
		Rectangle rect = sdLog.getVisibleRect();
		//sdLog.paintImmediately(sdLog.getVisibleRect());
		int a = sdLog.getScrollableUnitIncrement(rect, SwingConstants.VERTICAL, 1);
		sdLog.scrollRectToVisible(new Rectangle((int) rect.getX(), (int) rect.getY()+a, (int) rect.getWidth(), (int)rect.getHeight()));
		System.out.println(line);
	
	    }
	    sdLog.paintImmediately(sdLog.getVisibleRect());
	}
	catch (Exception e) { 
	    JOptionPane.showMessageDialog(this, "An error occured while trying to run executable\n"+e, "Error", JOptionPane.ERROR_MESSAGE);
	    System.out.println("[BRITE ERROR]: An error occured trying to run executable: " +e);
	    
	    BuildTopology.setEnabled(true);
	   
	    return;
	}
	
	BuildTopology.setEnabled(true);
       
    }
    

    public void processWindowEvent(WindowEvent e) {
	super.processWindowEvent(e);
	if (e.getID() == e.WINDOW_CLOSING){    
	    sd.dispose();
	    hPanel.dispose();
	    System.exit(0);
	}
    }


    String ROUTER_TOPOLOGY = "1 Level: ROUTER ONLY";
    String AS_TOPOLOGY = "1 Level: AS ONLY";
    String TOPDOWN_TOPOLOGY = "2 Level: TOP-DOWN";
    String BOTTOMUP_TOPOLOGY = "2 Level: BOTTOM-UP";
    String[] TopologyTypeData = {AS_TOPOLOGY, ROUTER_TOPOLOGY, TOPDOWN_TOPOLOGY, BOTTOMUP_TOPOLOGY};
    JComboBox TopologyType = new JComboBox(TopologyTypeData);
    
    String JAVAEXE = "Use Java Exe";
    String CPPEXE = "Use C++ Exe";
    String exeData[] = {CPPEXE, JAVAEXE};
    JComboBox ExeChoicesComboBox = new JComboBox(exeData);

    LineBorder lineBorder1 = new LineBorder(java.awt.Color.black);
    JLabel JLabel1 = new JLabel();
    
    JButton logo = new JButton(new ImageIcon("GUI"+System.getProperty("file.separator")+"images"+System.getProperty("file.separator")+
					   "brite4.jpg"));
    JButton BuildTopology = new JButton();
    JButton HelpButton = new JButton();

    StatusDialog sd = new StatusDialog(this);
  
    private Thread runThread = null;
    public Process p=null;
    
    JTabbedPane JTabbedPane1 = new JTabbedPane();
    ExportPanel ePanel = new ExportPanel();
    HelpPanel hPanel = new HelpPanel(this);  
    AboutPanel aboutPanel = new AboutPanel();
    ASPanel asPanel = new ASPanel();
    RouterPanel rtPanel= new RouterPanel();
    TDPanel tdPanel = new TDPanel(this);
    BUPanel buPanel = new BUPanel(this);
    boolean rtDisabled=false;
    boolean asDisabled=false;
    boolean hDisabled=true;
    
    public static void main(String args[])  {
	GUI.Brite g= new GUI.Brite();
	g.init();
	g.setVisible(true);
    }
}




final class ExportPanel  extends JPanel  implements ActionListener {
    
    EtchedBorder etchedBorder1 = new EtchedBorder();
    LineBorder lineBorder1 = new LineBorder(java.awt.Color.black);
    JLabel JLabel30 = new JLabel();
    JLabel JLabel31 = new JLabel();
    
    JCheckBox otterFormat, briteFormat;
    JLabel JLabel32 = new JLabel();
    JTextField ExportLocation = new JTextField();
    JButton ExportLocationBrowse = new JButton();
    JFileChooser fc = new JFileChooser();
    
    ExportPanel() { this.init(); } 
    
    public void actionPerformed(ActionEvent e) 
    { 
	if (e.getSource().equals(ExportLocationBrowse)) {
	    fc.showSaveDialog(this);
	    ExportLocation.setText(fc.getSelectedFile().getName());
	}
    }

    public boolean isBriteFormat() {
	return briteFormat.isSelected();
    }

    public boolean isOtterFormat() {
	return otterFormat.isSelected();
    }
    void init() {
	this.setBorder(etchedBorder1);
	this.setLayout(null);
	this.setBounds(24,370,432,96);
	JLabel30.setText("Export Topology");
	this.add(JLabel30);
	JLabel30.setForeground(java.awt.Color.black);
	JLabel30.setBounds(12,12,156,16);
	JLabel31.setText("File Format(s):");
	this.add(JLabel31);
	JLabel31.setForeground(java.awt.Color.black);
	JLabel31.setFont(new Font("SansSerif", Font.PLAIN, 12));
	JLabel31.setBounds(24,60,95,20);
	
	briteFormat = new JCheckBox("BRITE");
	briteFormat.setSelected(true);
	this.add(briteFormat);
	briteFormat.setFont(new Font("SansSerif", Font.PLAIN, 12));
	briteFormat.setBounds(132, 60, 72, 24);
	briteFormat.addActionListener(this);
	otterFormat = new JCheckBox("OTTER");
	this.add(otterFormat);
	otterFormat.setFont(new Font("SansSerif", Font.PLAIN, 12));
	otterFormat.setBounds(210,60,72,24);
	otterFormat.addActionListener(this);
	
	
	JLabel32.setText("Location:");
	this.add(JLabel32);
	JLabel32.setForeground(java.awt.Color.black);
	JLabel32.setFont(new Font("SansSerif", Font.PLAIN, 12));
	JLabel32.setBounds(24,36,96,20);
	ExportLocation.setBorder(lineBorder1);
	ExportLocation.setCursor(java.awt.Cursor.getPredefinedCursor(java.awt.Cursor.TEXT_CURSOR));
	this.add(ExportLocation);
	ExportLocation.setBounds(132,36,156,20);
	ExportLocationBrowse.setText("Browse...");
	ExportLocationBrowse.addActionListener(this);
	this.add(ExportLocationBrowse);
	ExportLocationBrowse.setFont(new Font("SansSerif", Font.PLAIN, 12));
	ExportLocationBrowse.setBounds(300,36,96,19);
	
    }
   
}


final class AboutPanel extends JDialog implements ActionListener {
    JEditorPane editPane;
    LineBorder lineBorder1 = new LineBorder(java.awt.Color.black);
    JButton closeB = new JButton();
        JScrollPane scrollPane1;

    public void actionPerformed(ActionEvent e) {
	if (e.getSource().equals(closeB)) {
	    setVisible(false);
	}
    }
    
    public void processWindowEvent(WindowEvent e) {
	super.processWindowEvent(e);
	if (e.getSource() == this && e.getID() == e.WINDOW_CLOSING) {
	  setVisible(false);
	}
    }
      
    public AboutPanel() {
	super();
	super.dialogInit();
	setSize(300,300);
	setResizable(false);
	
	
	getContentPane().setLayout(null);
	getContentPane().setBackground(new java.awt.Color(204,204,204));
	
	getContentPane().add(closeB);
	closeB.setBounds(80, 260, 100, 21);
	closeB.setText("Close Window");
	closeB.setFont(new Font("SansSerif", Font.PLAIN, 10));
	closeB.setBorder(lineBorder1);
	closeB.addActionListener(this);
	closeB.setVisible(true);
	       
	editPane = new JEditorPane();
	String s= null;
	try {
	  s = "file:"+System.getProperty("user.dir") + System.getProperty("file.separator") + "GUI"+
	    System.getProperty("file.separator")+"help"+ System.getProperty("file.separator")+ "about.html";
	  
	  editPane.setPage(new java.net.URL(s));
	}
	catch (Exception e) {
	  System.out.println("[BRITE ERROR] Could not read about file " + e);
	}
	editPane.setEditable(false);
	editPane.setBounds(10,10,280,240);
	scrollPane1 = new JScrollPane(editPane, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	getContentPane().add(scrollPane1);
	scrollPane1.setBounds(10, 10, 280, 240);
	
	setTitle("About BRITE");
	setSize(getPreferredSize());
    }



}




final class HelpPanel extends JDialog  implements ActionListener{
    JButton closeB = new JButton("Close Help Window");
 
    JScrollPane scrollPane1;
    JEditorPane editPane;
    LineBorder lineBorder1 = new LineBorder(java.awt.Color.black);
    GUI.Brite parent =null;
    
    public void actionPerformed(ActionEvent e) {
	if (e.getSource().equals(closeB)) {
	  setVisible(false);
	}
    }

    public void processWindowEvent(WindowEvent e) {
	super.processWindowEvent(e);
	if (e.getSource() == this && e.getID() == e.WINDOW_CLOSING) {
	  setVisible(false);
	}
    }
   
    public JScrollPane getScroll() { return scrollPane1; }
    public JButton getButton() { return closeB; }
    public HelpPanel(GUI.Brite parent) {
	super();
	super.dialogInit();
	setSize(500,500);
	setResizable(false);
	
	this.parent = parent; //we need this because sometimes we need to kill the process while its executing
	
	getContentPane().setLayout(null);
	getContentPane().setBackground(new java.awt.Color(204,204,204));
	
	getContentPane().add(closeB);
	closeB.setBounds(200, 470, 100, 21);
	closeB.setText("Close Window");
	closeB.setFont(new Font("SansSerif", Font.PLAIN, 10));
	closeB.setBorder(lineBorder1);
	closeB.addActionListener(this);
	closeB.setVisible(true);
	       
	editPane = new JEditorPane();
	String s= null;
	try {
	  s = "file:"+System.getProperty("user.dir") + System.getProperty("file.separator") + "GUI"+
	    System.getProperty("file.separator")+"help"+ System.getProperty("file.separator")+ "parameterhelp.html";
	  
	  editPane.setPage(new java.net.URL(s));
	}
	catch (Exception e) {
	  System.out.println("[BRITE ERROR] Could not read help file " + e);
	
	}
	editPane.setEditable(false);
	editPane.setBounds(10,10,480,450);
	scrollPane1 = new JScrollPane(editPane, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	getContentPane().add(scrollPane1);
	scrollPane1.setBounds(10, 10, 480, 450);
	
	setTitle("BRITE Help ");
	setSize(getPreferredSize());
    }


} 









final class StatusDialog extends JDialog implements ActionListener {
  JButton closeB = new JButton("Close Status Window");
  JButton cancelB = new JButton("Cancel Generation");
  JTextArea statusText = new JTextArea();
    JScrollPane scrollPane1;
    LineBorder lineBorder1 = new LineBorder(java.awt.Color.black);
  GUI.Brite parent =null;

  
    public void actionPerformed(ActionEvent e) {
	if (e.getSource().equals(closeB)) {
	    statusText.setText("");
	    setVisible(false);
	}
	if (e.getSource().equals(cancelB)) {
	  parent.p.destroy();
	  statusText.append("*** Generation Cancelled by user. ***");

	}
    }

    public void processWindowEvent(WindowEvent e) {
	super.processWindowEvent(e);
	if (e.getSource() == this && e.getID() == e.WINDOW_CLOSING) {
	    statusText.setText("");
	    setVisible(false);
	}
    }

       
    public JTextArea getTextArea() { return statusText; }
    public JScrollPane getScroll() { return scrollPane1; }
    public JButton getButton() { return closeB; }
    public StatusDialog(GUI.Brite parent) {
	super();
	super.dialogInit();
	setSize(400,300);
	setResizable(false);
	
	this.parent = parent; //we need this because sometimes we need to kill the process while its executing
	
	getContentPane().setLayout(null);
	getContentPane().setBackground(new java.awt.Color(204,204,204));
	
	getContentPane().add(closeB);
	closeB.setBounds(200, 170, 100, 21);
	closeB.setText("Close Window");
	closeB.setFont(new Font("SansSerif", Font.PLAIN, 10));
	closeB.setBorder(lineBorder1);
	closeB.addActionListener(this);
	closeB.setVisible(true);
	
	getContentPane().add(cancelB);
	cancelB.setBounds(75,170, 100, 21);
	cancelB.setText("Cancel Generation");
	cancelB.setFont(new Font("SansSerif", Font.PLAIN, 10));
	cancelB.addActionListener(this);
	cancelB.setBorder(lineBorder1);
	cancelB.setVisible(true);
       
	
	statusText.setFont(new Font("SansSerif", Font.PLAIN, 10));
	statusText.setBounds(10,10,380,150);
	statusText.setLineWrap(true);
	statusText.setEditable(false);
	
	scrollPane1 = new JScrollPane(statusText, ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS, ScrollPaneConstants.HORIZONTAL_SCROLLBAR_AS_NEEDED);
	getContentPane().add(scrollPane1);
	scrollPane1.setBounds(10, 10, 380, 150);
	//	scrollPane1.getViewport().setScrollMode(JViewport.BLIT_SCROLL_MODE);
	
	setTitle("Status Window");
	setSize(getPreferredSize());
    }

  
  

}




