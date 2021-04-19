function helloworld()
    println("Hello, World!")
end

function readfile(filename::String)
    io = open(filename,"r")
        println(read(s,String))
    end
end
#helloworld()
readfile("dummy.txt")

