function stringify(str) {
    return JSON.stringify((str))
}
function parse(str) {
    return JSON.parse((str))
}

module.exports = { parse,stringify }