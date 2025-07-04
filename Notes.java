Excepption handlling::  11th July.

Types of errors::

  System generated error -> compile time (Due to syntax)
                         -> Runtime - ClassNotFound, FileNotFound, AIOB, SOF...
                         -> Logical - Calculations, incorrect values, etc
  Exception:: Converting System generated error to user-understandable message.                
          Structure / Hierarchy
          
          Object
          Throwable
          -Error
            -StackOverFlow Error
            
            
            
            
        --> Handling the exception is just converting system generated error message into user-friendly error message. Whenever exception occurs, JVM creates an object of appropriate exception of subclass & generates a system error message. This system generated messages aren't understandable by user. Therefore, we convert it to user-friendly error message. This is possible using : 
        (i) try{} - writing risky code
            catch{} - handles the exception: the cleanup activity code is written.
        (ii) finally{} - also does the cleanup activity too. 
                         There are three tyes: final, finally( database conn, Server conn, (Closing up activity code)), finalizer() - Written with respect to object.
        (iii) throw,throws - when user wants to send an exception. They're just keywords and not block of codes.
        Eg. Try block ::
        try{
          Statement-1;
          Statement-2;
          Statement-3;
  *    }
        catch(Exception e){
          Statement-4;
        }
        Statement-5;
        -> If there's no exception, Statements 1,2,3&5 will be executed. | Normal termination
        -> In-case of an exception, say at statement-2 & corresponding catch block is matched. Statement 1,4 and 5 will be executed. | Normal termination. 
        -> In-case of exception with Statement-2, but the catch block hasn't matched, it's an abnormal termination. Statement 4 won'e be executed either.
      --> Conclusions:: Whithin try block, if anywhere exception is raised, then rest of the try blocks won't be executed even tho we handle it. Hence, length of try block should be as less as possible. and we have to put risky code alone within it. In addition to try block, there's a chance to raise exception in catch/finally blocks. if any statement raises an exception, that isn't part of try block, then it is always an abnormal situation. 
      Eg.2: : 
      try{
          Statement-1;
          Statement-2;
          Statement-3;
  *    }
        catch(Exception e){
          Statement-4;
        }
        finally{
          Statement-5;
        }
        Statement-6;
        
        -> if there's no exception, 1,2,3,5,6 will be executed. | Normal termination.
        -> if there's exception in 2, & catch block is matched, 1,4,5,6 will be executed. | Normal termination
        -> catch block isn't matched. 1,5 will be executed . | Abormal termination & Statement 6 won't be executed.
        -> if exception is at 4, we can't handle the exception. 1,2,3,5 will be executed. | Abnormal termination
        -> if exception is at 5 or 6, it's an abnormal termination.
  --> There can be more than one catch block of one try block.
  --> Whenever We're writing catch block, we should i.e, try alone without catch/finally is syntax error.
  --> While writing catch block or finally, we HaveTo write try block, meaning catch without try is invalid.
  --> In try,catch/finally, the order is important.
  --> try with multiplt catch blocks is valid, but the order is important.
  --> If we're defining two catch blocks for the same exception, we'll get compile time error.
  --> We can define try,catch,finally within a try,catch or finally block. i.e, a nested block.
  --> For try,catch/finally, curly braces are mandatory.
  
  
__________________________
Multithreading - UNIT IV

newborn state - thread is created.
Runable state - started | pricess is still not allocated.
Running state - run | active and under execution.
Blocked thread - inactive state | sleep/wait mode
Dead state - stopped.

Life Cycle of thread:::

