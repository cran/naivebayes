naive_bayes <- function(x, ...) {
    UseMethod("naive_bayes")
}


naive_bayes.default <- function(x, y, prior = NULL, laplace = 0,
                                usekernel = FALSE, ...) {

    data <- as.data.frame(x)
    levels <- sort(as.character(unique(y)))
    vars <- names(data)

    if (!is.factor(y) && !is.character(y) && !is.logical(y))
        stop("y has to be either a factor or character or logical vector")

    if (is.factor(y) && nlevels(y) != length(levels)) {
        warning("Number of unique values in the class variable is not equal to number of levels")
        y <- as.character(y)
    }

    if (is.null(prior)) {
        prior <- prop.table(table(y, dnn = ""))
    } else {
        if (length(prior) != length(levels))
            stop(paste0("Vector with prior probabilities should have ",
                        length(levels), " entries"))
        prior <- stats::setNames(prior / sum(prior), levels)
    }

    tables <- sapply(names(data), function(x) {
        var <- data[[x]]
        if (is.numeric(var)) {
            if (usekernel) {
                tapply(var, y, function(x, ...) stats::density(x, na.rm = TRUE, ...))
            } else {
                tab <- rbind(tapply(var, y, mean, na.rm = TRUE),
                             tapply(var, y, stats::sd, na.rm = TRUE))
                rownames(tab) <- c("mean", "sd")
                names(dimnames(tab)) <- c(x, "")
                as.table(tab)
            }
        } else {
            tab <- table(y, var, dnn = c("", x))
            t((tab + laplace) / (rowSums(tab) + laplace * nrow(tab)))
        }
    }, simplify = FALSE)

    structure(list(data = list(x = data, y = y), levels = levels,
                   laplace = laplace, tables = tables, prior = prior,
                   usekernel = usekernel, call = match.call()),
              class = "naive_bayes")
}


naive_bayes.formula <- function(formula, data, prior = NULL, laplace = 0,
                                usekernel = FALSE,  subset,
                                na.action = stats::na.pass, ...) {

    mf <- match.call(expand.dots = FALSE)
    mf[c("prior", "laplace", "usekernel", "...")] <- NULL
    mf$na.action <- na.action
    mf[[1]] <- as.symbol("model.frame")
    data <- eval.parent(mf)
    y_name <- names(data)[1]
    y <- stats::model.response(data)
    data <- data[-1]

    if (!is.factor(y) && !is.character(y) && !is.logical(y))
        stop("y has to be either a factor or character or logical vector")

    res <- naive_bayes.default(data, y, prior, laplace, usekernel, ...)
    res$call <- match.call()
    res
}


predict.naive_bayes <- function(object, newdata = NULL, type = c("class", "prob"),
                                threshold = 0.001, ...) {

    if (is.null(newdata)) newdata <- object$data$x
    else newdata <- as.data.frame(newdata)
    na <- sapply(newdata, anyNA)
    type <- match.arg(type)
    lev <- object$levels
    n_lev <- length(lev)
    n_obs <- dim(newdata)[1L]
    usekernel <- object$usekernel
    prior <- as.double(object$prior)
    tables <- object$tables
    features <- names(newdata)[names(newdata) %in% names(tables)]
    log_sum <- 0

    for (var in features) {
        V <- newdata[[var]]
        if (is.numeric(V)) {
            tab <- tables[[var]]
            if (usekernel) {
                p <- sapply(lev, function(z) {
                    dens <- tab[[z]]
                    stats::approx(dens$x, dens$y, xout = V, rule = 2)$y
                })
                p[p == 0] <- threshold
                if (na[var]) p[is.na(p)] <- 1
                log_sum <- log_sum + log(p)
            } else {
                dimnames(tab) <- NULL
                s <- tab[2, ]
                s[s == 0] <- threshold
                p <- sapply(seq_along(lev), function(z) {
                    stats::dnorm(V, tab[1, z], s[z])
                })

                p[p == 0] <- threshold

                if (na[var]) p[is.na(p)] <- 1

                log_sum <- log_sum + log(p)
            }

        } else {
            tab <- tables[[var]]
            if (class(V) == "logical") V <- as.character(V)
            if (na[var]) {
                na_ind <- which(is.na(V))
                V[na_ind] <- attributes(tab)$dimnames[[1]][1]
                p <- tab[V, ]
                if (n_obs == 1) p <- 1
                else p[na_ind, ] <- 1
            } else {
                p <- tab[V, ]
            }
            log_sum <- log_sum + log(p)
        }
    }
    if (type == "class") {
        if (n_obs == 1) {
            post <- log_sum + log(prior)
            return(factor(lev[which.max(post)], levels = lev))
        } else {
            post <- t(t(log_sum) + log(prior))
            return(factor(lev[max.col(post, "first")], levels = lev))
        }
    } else {
        if (n_obs == 1) {
            lik <- exp(log_sum + log(prior))
            post <- sapply(lik, function(prob) {
                prob / sum(lik)
            })
            return(t(as.matrix(post)))
        } else {
            lik <- exp(t(t(log_sum) + log(prior)))
            dimnames(lik) <- NULL
            rs <- rowSums(lik)
            colnames(lik) = lev
            return(apply(lik, 2, function(prob) {
                prob / rs
            }))
        }
    }
}


print.naive_bayes <- function(x, ...) {
    cat("===================== Naive Bayes =====================", "\n")
    cat("Call:", "\n")
    print(x$call)
    cat("\n")
    cat("A priori probabilities:", "\n")
    print(x$prior)
    cat("\n")
    cat("Tables:", "\n")
    n <- length(x$tables)
    for (i in 1:n) {
        if (i >= 6) next
        print(x$tables[[i]])
        cat("\n")
    }
    if (n > 5) cat("# ... and", n - 5, "more tables")
}


plot.naive_bayes <- function(x, which = NULL, ask = FALSE, legend = TRUE,
                             legend.box = FALSE, arg.num = list(),
                             arg.cat = list(), ...) {

    vars <- names(x$tables)

    if (is.character(which) && !all(which %in% vars))
        stop("At least one variable is not available")

    if (length(which) > length(vars))
        stop("Too many variables selected")

    if (!is.null(which) && !is.character(which) && !is.numeric(which))
        stop("'which' has to be either character or numeric vector")

    if (length(list(...)) > 0)
        warning("Please specify additional parameters with 'arg.num' or 'arg.cat'")

    if (is.null(which))
        which <- seq_along(vars)

    if (is.numeric(which))
        v <- vars[which]

    if (is.character(which))
        v <- vars[vars %in% which]

    opar <- graphics::par()$ask
    graphics::par(ask = ask)
    on.exit(graphics::par(ask = opar))

    for (i in v) {
        i_tab <- x$tables[[i]]
        lev <- x$levels
        if (is.numeric(x$data$x[[i]]))  {
            if (x$usekernel) {
                bws <- round(sapply(i_tab, "[[", "bw"), 3)
                leg <- paste0(lev, " (bw: ", bws, ")")
                X <- sapply(i_tab, "[[", "x")
                Y <- sapply(i_tab, "[[", "y")
            }
            if (!x$usekernel) {
                leg <- lev
                r <- range(x$data$x[[i]])
                X <- seq(r[1], r[2], length.out = 512)
                Y <- matrix(stats::dnorm(x = X,
                                         mean = rep(i_tab[1, ], each = length(X)),
                                         sd   = rep(i_tab[2, ], each = length(X))),
                            ncol = length(lev))
            }
            n <- names(arg.num)
            if (!("col"  %in% n)) arg.num$col <- seq_along(lev) + 1
            if (!("type" %in% n)) arg.num$type <- "l"
            if (!("lty"  %in% n)) arg.num$lty <- seq_along(lev)
            if (!("ylab" %in% n)) arg.num$ylab <- "Density"
            arg.num$xlab <- i

            params <- c(list(x = quote(X), y = quote(Y)), arg.num)
            do.call("matplot", params)
            if (legend) {
                bty = ifelse(legend.box == TRUE, TRUE, "n")
                legend("topleft", leg, col = arg.num$col, lty = arg.num$lty,
                       title = "", cex = 1, y.intersp = 0.75, bty = bty)
            }

        } else {
            if (!("main" %in% names(arg.cat))) arg.cat$main <- ""
            if (!("color" %in% names(arg.cat))) arg.cat$color <- seq_along(lev) + 1
            arg.cat$xlab <- i
            params <- c(list(x = quote(i_tab)), c(arg.cat))
            do.call("mosaicplot", params)
        }
    }
    invisible()
}


tables <- function(object, which = NULL) {

    vars <- names(object$tables)

    if (is.character(which) && !all(which %in% vars))
        stop("At least one variable is not available")

    if (length(which) > length(vars))
        stop("too many variables selected")

    if (!is.null(which) && !is.character(which) && !is.numeric(which))
        stop("'which' has to be either character or numeric vector")

    if (is.null(which))
        which <- seq_along(vars)

    if (is.numeric(which))
        v <- vars[which]

    if (is.character(which))
        v <- vars[vars %in% which]

    object$tables[v]
}
