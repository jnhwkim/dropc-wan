ifeq ($(dbg),1)
	FLAG:=-DCMAKE_BUILD_TYPE:STRING=Debug 
	#FLAG:=-DCMAKE_BUILD_TYPE=Debug 
	TARGET:=debug
else
ifeq ($(dbg),0)
	FLAG:=-DCMAKE_BUILD_TYPE:STRING=ReleaseO1
	#FLAG:=-DCMAKE_BUILD_TYPE=Debug 
	TARGET:=build
else
	FLAG:=-DCMAKE_BUILD_TYPE:STRING=Release 
	#FLAG:=-DCMAKE_BUILD_TYPE=Release 
	TARGET:=build
endif
endif

testFile0 := test_dropc
testFile1 := test_dropc_bit

all:
	mkdir -p $(TARGET)
	cd $(TARGET) && cmake $(FLAG) ..
	cd $(TARGET) && make -f Makefile
	ln -sf $(TARGET)/$(testFile0)
	ln -sf $(TARGET)/$(testFile1)

clean:
	rm -rf $(TARGET)
	rm -f $(testFile0)
	rm -f $(testFile1)
	rm -f $(testFile2)
	rm -rf test_result
