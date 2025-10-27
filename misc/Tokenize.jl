module Tokenize
using DataStructures:Trie
using StringEncodings:decode
using StatsBase:countmap
export load_text_file, pre_tokenize, find_unique_chars, find_unique_words, 
        TokenDict, tokenize, 
        MostFrequent, WordPieceScore, bpe

function pre_tokenize(s::AbstractString, skipdigits = false) 
    if !skipdigits
        split(s, r"[^A-Za-z0-9]+",keepempty=false)
    else
        split(s, r"[^A-Za-z]+",keepempty=false)
    end
end

function find_unique_chars(words)   #words::AbstractVector{<:AbstractString})
    union(Set{Char}(),words...)
end

function find_unique_words(quora; sorted::Bool=false)   #quora::AbstractVector{<:AbstractString}
    words = Set{AbstractString}()
    for line in quora
        uwl = unique(pre_tokenize(line))
        union!(words, uwl)
    end
    sorted ? words |> collect |> sort! : words
end

function load_text_file(filename::AbstractString; splitlines::Bool=true, pretokenize::Bool=true, skipdigits::Bool=true)
    open(filename, "r") do file
        content = read(file)
        content = decode(content, "UTF-8")  # Replace "UTF-8" with your file's encoding if different
        content = filter(x-> x ∉ "\r:.", content)
        lines = (splitlines ? split(content, "\n", keepempty = false) : content)
        if pretokenize
            return collect(Iterators.flatten([pre_tokenize(l,skipdigits) for l in lines]))
        else
            return lines
        end
    end
end

function count_occurences(words, tokens)
    cnt = Dict{String,Int}()
    for tok in tokens
        cnt[tok] = sum( [length(split(w,tok)) for w in words] )
    end
    cnt
end

mutable struct TokenDict
    token_to_id::Dict{String,Int}
    id_to_token::Dict{Int,String}
    token_freq::Dict{String,Int}
    next_token_id::Int
    trie::Union{Trie{Char,Int},Nothing}
end

function TokenDict(words)::TokenDict
    unique_chars = sort!(collect(find_unique_chars(words)))
    token_to_id = Dict{String,Int}();
    id_to_token = Dict{Int,String}();
    sizehint!(token_to_id,length(unique_chars));
    sizehint!(id_to_token,length(unique_chars));
    for (i,c) in enumerate(unique_chars)
        tok = string(c)
        token_to_id[tok] = i;
        id_to_token[i] = tok;
    end
    next_token_id = length(token_to_id) + 1
    token_freq = count_occurences(words, keys(token_to_id))

    TokenDict(token_to_id,id_to_token,token_freq,next_token_id,nothing)
end

function build_trie(td::TokenDict)::Trie{Char,Int}
    td.trie = Trie{Char,Int}(td.token_to_id)
end

#works only for single char tokens for now
function tokenize(str::AbstractString, td::TokenDict)::Vector{Vector{Int}}
    [[td.token_to_id[string(char)] for char in word] for word in pre_tokenize(str)]
end

function tokenize(vec, td::TokenDict)::Vector{Vector{Int}}
    collect(Iterators.flatten([[[td.token_to_id[string(char)] for char in word] for word in pre_tokenize(str)] for str in vec]))
end

struct SearchState
    tokens::Vector{Int}
    pos::Int
end

function tokenize_word(str::AbstractString, tr::Trie{Char,Int})::Vector{Int}
    stack = [ SearchState([],1) ]
    best = []
    N = length(str)
    N_best = N
    while !isempty(stack)
        ss = pop!(stack)
        i = ss.pos
        for j in i:N
            id = get(tr,str[i:j],nothing)
            if id !== nothing
                tokens = [ss.tokens...,id]
                if j < N
                    push!(stack,SearchState(tokens,j+1))
                else
                    nl = length(tokens)
                    if nl < N_best
                        best = [tokens]
                        N_best = nl
                    elseif nl == N_best
                        push!(best,tokens)
                    end
                end
            end
        end
    end
    best[1] #how to choose among multiple best tokenizations?
end

function tokenize(words, tr::Trie{Char,Int})
    [ tokenize_word(w,tr) for w in words]
end

function append!(td::TokenDict, tok::String)
    if tok ∉ keys(td.token_to_id)
        tid = td.next_token_id
        td.token_to_id[tok] = tid
        td.id_to_token[tid] = tok
        td.next_token_id += 1
        true
    else
        false
    end
end

function append!(td::TokenDict, vec::Vector{String})
    for tok in vec
        append!(td,tok)
    end
end

function new_token_from_pair(td::TokenDict, pair::Tuple{Int,Int})
    tok = td.id_to_token[pair[1]] * td.id_to_token[pair[2]]
    tid = td.next_token_id
    td.token_to_id[ tok ] = tid
    td.id_to_token[ tid ] = tok
    td.next_token_id += 1
    tok,tid
end

function replace_pairs(vec::Vector{Int}, pair::Tuple{Int,Int}, id::Int)
    result = Vector{Int}()
    i=1
    while i <= length(vec)
        if i < length(vec) && vec[i] == pair[1] && vec[i+1] == pair[2]
            push!(result, id)
            i += 2
        else
            push!(result, vec[i])
            i += 1
        end
    end
    result
end

function replace_pairs(vec::Vector{Vector{Int}}, pair::Tuple{Int,Int}, nid::Int)
    [ replace_pairs(v, pair, nid) for v in vec ]
end

abstract type PairEvalPolicy end
struct MostFrequent <: PairEvalPolicy end
struct WordPieceScore <: PairEvalPolicy end

function pair_eval(pairs::Vector{Tuple{Int,Int}}, td::TokenDict, p::MostFrequent)
    cnt = countmap(pairs)
    freq,pair = findmax(cnt)
    pair,freq
end

function pair_eval(pairs::Vector{Tuple{Int,Int}}, td::TokenDict, p::WordPieceScore)
    cnt = countmap(pairs)
    tf = td.token_freq
    i2t = td.id_to_token
    score = Dict(p => cnt[p] / (tf[i2t[p[1]]] * tf[i2t[p[2]]]) for p in pairs)
    freq,pair = findmax(score)
    pair,cnt[pair]
end

function get_pairs(vec::Vector{Int})::Vector{Tuple{Int,Int}}
    [(vec[i], vec[i+1]) for i in 1:length(vec)-1]
end

function get_pairs(vec::Vector{Vector{Int}})::Vector{Tuple{Int,Int}}
    collect(Iterators.flatten([get_pairs(v) for v in vec]))
end

function bpe(words::AbstractVector{<:AbstractString}, n_iter::Union{Int,Nothing}=nothing; eval_policy::PairEvalPolicy = MostFrequent(), start_token::String = "<S>", end_token::String = "<E>")::TokenDict
    token_dict = TokenDict(words)
    append!(token_dict,[start_token,end_token])
    tokenized_quora = tokenize(words, token_dict)

    if isnothing(n_iter)
        n_iter = typemax(Int64) # set to max value for Int64
    end

    for _ in 1:n_iter
        pair,freq = pair_eval(get_pairs(tokenized_quora), 
                                    token_dict, eval_policy)
        nt,new_token_id = new_token_from_pair(token_dict,pair)
        t1,t2 = token_dict.id_to_token[pair[1]], token_dict.id_to_token[pair[2]]
        token_dict.token_freq[nt] = freq
        new_tokenized_quora = replace_pairs(tokenized_quora, pair, new_token_id)
        if all( v->length(v) == 1, new_tokenized_quora)
            break
        else
            tokenized_quora = new_tokenized_quora
        end
    end
    
    token_dict
end

end #module Tokenize