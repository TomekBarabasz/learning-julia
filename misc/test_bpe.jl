include("Tokenize.jl")
using .Tokenize
using Test

words = ["tomek", "asia", "marcin"]
join_w_comma(v) = join(v, ", ")

n_iter = length(ARGS) >=1  ? parse(Int, ARGS[1]) : 10
policy = begin
    if length(ARGS) < 2
        MostFrequent()
    else
        ARGS[2] == "wp" ? WordPieceScore() : MostFrequent()
    end
end
td = bpe(words, n_iter; eval_policy=policy)
get_tokens(td) |> join_w_comma |> println
tr = build_trie(td)
tokens = tokenize(tr, words)
for (w,tok) in zip(words,tokens)
    println("word: '$w' => [ $(join_w_comma(tok)) ]")
end
