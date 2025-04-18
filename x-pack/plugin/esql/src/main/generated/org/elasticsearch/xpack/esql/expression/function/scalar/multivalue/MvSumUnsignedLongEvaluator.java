// Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
// or more contributor license agreements. Licensed under the Elastic License
// 2.0; you may not use this file except in compliance with the Elastic License
// 2.0.
package org.elasticsearch.xpack.esql.expression.function.scalar.multivalue;

import java.lang.ArithmeticException;
import java.lang.Override;
import java.lang.String;
import org.elasticsearch.compute.data.Block;
import org.elasticsearch.compute.data.LongBlock;
import org.elasticsearch.compute.operator.DriverContext;
import org.elasticsearch.compute.operator.EvalOperator;
import org.elasticsearch.compute.operator.Warnings;
import org.elasticsearch.xpack.esql.core.tree.Source;

/**
 * {@link EvalOperator.ExpressionEvaluator} implementation for {@link MvSum}.
 * This class is generated. Edit {@code MvEvaluatorImplementer} instead.
 */
public final class MvSumUnsignedLongEvaluator extends AbstractMultivalueFunction.AbstractNullableEvaluator {
  private final Source source;

  private Warnings warnings;

  public MvSumUnsignedLongEvaluator(Source source, EvalOperator.ExpressionEvaluator field,
      DriverContext driverContext) {
    super(driverContext, field);
    this.source = source;
  }

  @Override
  public String name() {
    return "MvSum";
  }

  /**
   * Evaluate blocks containing at least one multivalued field.
   */
  @Override
  public Block evalNullable(Block fieldVal) {
    LongBlock v = (LongBlock) fieldVal;
    int positionCount = v.getPositionCount();
    try (LongBlock.Builder builder = driverContext.blockFactory().newLongBlockBuilder(positionCount)) {
      for (int p = 0; p < positionCount; p++) {
        int valueCount = v.getValueCount(p);
        if (valueCount == 0) {
          builder.appendNull();
          continue;
        }
        try {
          int first = v.getFirstValueIndex(p);
          int end = first + valueCount;
          long value = v.getLong(first);
          for (int i = first + 1; i < end; i++) {
            long next = v.getLong(i);
            value = MvSum.processUnsignedLong(value, next);
          }
          long result = value;
          builder.appendLong(result);
        } catch (ArithmeticException e) {
          warnings().registerException(e);
          builder.appendNull();
        }
      }
      return builder.build();
    }
  }

  private Warnings warnings() {
    if (warnings == null) {
      this.warnings = Warnings.createWarnings(
              driverContext.warningsMode(),
              source.source().getLineNumber(),
              source.source().getColumnNumber(),
              source.text()
          );
    }
    return warnings;
  }

  public static class Factory implements EvalOperator.ExpressionEvaluator.Factory {
    private final Source source;

    private final EvalOperator.ExpressionEvaluator.Factory field;

    public Factory(Source source, EvalOperator.ExpressionEvaluator.Factory field) {
      this.source = source;
      this.field = field;
    }

    @Override
    public MvSumUnsignedLongEvaluator get(DriverContext context) {
      return new MvSumUnsignedLongEvaluator(source, field.get(context), context);
    }

    @Override
    public String toString() {
      return "MvSum[field=" + field + "]";
    }
  }
}
