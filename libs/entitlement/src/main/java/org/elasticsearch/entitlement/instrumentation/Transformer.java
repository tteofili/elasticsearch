/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.entitlement.instrumentation;

import java.lang.instrument.ClassFileTransformer;
import java.security.ProtectionDomain;
import java.util.Set;

/**
 * A {@link ClassFileTransformer} that applies an {@link Instrumenter} to the appropriate classes.
 */
public class Transformer implements ClassFileTransformer {
    private final Instrumenter instrumenter;
    private final Set<String> classesToTransform;

    private boolean verifyClasses;

    public Transformer(Instrumenter instrumenter, Set<String> classesToTransform, boolean verifyClasses) {
        this.instrumenter = instrumenter;
        this.classesToTransform = classesToTransform;
        this.verifyClasses = verifyClasses;
        // TODO: Should warn if any MethodKey doesn't match any methods
    }

    public void enableClassVerification() {
        this.verifyClasses = true;
    }

    @Override
    public byte[] transform(
        ClassLoader loader,
        String className,
        Class<?> classBeingRedefined,
        ProtectionDomain protectionDomain,
        byte[] classfileBuffer
    ) {
        if (classesToTransform.contains(className)) {
            // System.out.println("Transforming " + className);
            return instrumenter.instrumentClass(className, classfileBuffer, verifyClasses);
        } else {
            // System.out.println("Not transforming " + className);
            return null;
        }
    }

    // private static final Logger LOGGER = LogManager.getLogger(Transformer.class);
}
