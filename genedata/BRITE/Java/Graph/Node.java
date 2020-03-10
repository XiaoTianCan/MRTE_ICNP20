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

package Graph;


import java.util.Comparator;


/**
   The Node class for our Graph contains only the base
   minimum member variables.  Environment specific semantics (such as Router
   or AS etc) that are often attached to a node are divorced from the
   Node class and stored in a decorator class, NodeConf (short for
   NodeConfiguration).  As such you can add/remove attributes to the Node at
   run-time without having to change the Node representation.  
   <p>
   We use the decorator pattern as the relationship between a Node and
   its configuration, NodeConf.  This pattern is described in Design Patterns:
   Elements of Reusable Object-Oriented Software by Gamma et al.
   ISBN#: 0-201-63361-2.
   <p>
   Unique NodeIDs (used in Graph) are computed by maintaining a static 
   variable that is incremented each time a new nodes is created.  
*/
public final class Node {

    int id;
    int addr;
    int indegree;
    int outdegree;
    int color;
    NodeConf n; 

    static int nodeCount=-1;

    /** provides a comparator to sort nodes by their IDs*/
    public static NodeIDComparator IDcomparator = new NodeIDComparator();
    
    /** Class Constructor1.  Assigns the node a unique ID (maintained
	by a static var) and does other initialization.  */
    public Node() { 
	this.id = ++nodeCount;
	this.color = GraphConstants.COLOR_WHITE;
	indegree=0;
	outdegree=0 ;
    }

    
    /** Constructor2: same as Constructor 1 but allows convenience of
	specifying indegree and outdegree of the nodes.  Useful when
	importing other topologies. */
    public Node(int inDeg, int outDeg) {
	this.id=++nodeCount;
	this.color=GraphConstants.COLOR_WHITE;
	this.indegree= inDeg;
	this.outdegree = outDeg;
	
    }
    
    /*get methods*/
    public int getID() { return id; }
    public int getAddr() { return addr; }
    public int getInDegree() { return indegree;}
    public int getOutDegree() { return outdegree; }
    public int getColor() { return color;}
    public NodeConf getNodeConf() { return n; }
    public static int getNodeCount() { return nodeCount; }
    public static int getUniqueID() { ++nodeCount; return nodeCount; }

    /*set methods*/
    public void setID(int id) { this.id = id;}
    public void setAddr(int addr) { this.addr = addr;}
    public void setInDegree(int indegree) { this.indegree = indegree; }
    public void setOutDegree(int outdegree) { this.outdegree=outdegree; }
    public void setColor(int c) { this.color = c;}
    public void setNodeConf(NodeConf n) { this.n = n;}
    
    public void incrementInDegree() { ++this.indegree; }
    public void incrementOutDegree() { ++this.outdegree; } 


}



/** NodeID comparator provides a comparator to compare Node IDs.  You
 can follow this template and trivially write your own comparator if
 you need for instance, to sort the nodes in another fashion, eg
 indegrees.  We use this comparator to sort nodes when printing them
 to a file.*/
class NodeIDComparator implements Comparator {
    
    public int compare(Object n1, Object n2) {
	int n1id = ((Node)n1).getID();
	int n2id = ((Node)n2).getID();
	
	if (n1id  < n2id) return -1;
	if (n1id == n2id) return 0;
	if (n1id  > n2id) return 1;
	
	return 1;  //never gets here but javac complains
    }

}





