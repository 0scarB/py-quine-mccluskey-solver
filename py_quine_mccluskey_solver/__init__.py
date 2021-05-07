from dataclasses import dataclass
import re
from typing import cast, Callable, Optional


TRow = list[str]
TTable = list[TRow]


def remove_columns(table: TTable, column_indices: list[int]) -> TTable:
    return [
        [
            cell
            for column_index, cell in enumerate(row)
            if column_index not in column_indices
        ]
        for row in table
    ]


def solve(
    table: TTable,
    output_column_indices: Optional[list[int]] = None,
    as_latex: bool = False,
) -> list[str]:
    columns_n = len(table[0])

    if output_column_indices is None:
        output_column_indices = [columns_n - 1]

    solutions: list[str] = []

    for output_column_index in output_column_indices:
        new_column_indices_original_column_indices_map = [
            column_index
            for column_index in range(columns_n)
            if column_index == output_column_index
            or column_index not in output_column_indices
        ]

        solutions.append(
            solve_for_output_column(
                table=remove_columns(
                    table,
                    [
                        output_column_index_to_remove
                        for output_column_index_to_remove in output_column_indices
                        if output_column_index_to_remove != output_column_index
                    ],
                ),
                output_column_index=new_column_indices_original_column_indices_map.index(
                    output_column_index
                ),
                as_latex=as_latex,
            )
        )

    return solutions


def solve_for_output_column(
    table: TTable, output_column_index: int, as_latex: bool = False
) -> str:
    column_names = table[0]

    minterm_table = get_table_with_min_and_dont_care_terms(
        table[1:], output_column_index=output_column_index
    )

    implicants = convert_minterm_table_to_implicants(
        minterm_table, output_column_index=output_column_index
    )

    implicants_n = len(implicants)

    prime_implicants = []

    while len(implicants) > 0:
        sorted_implicants, group_ranges = get_sorted_implicants_and_group_ranges(
            implicants
        )

        implicants, new_prime_implicants = get_combined_implicants_and_prime_implicants(
            implicants=sorted_implicants, group_ranges=group_ranges
        )

        prime_implicants += new_prime_implicants

    essential_prime_implicants = get_essential_prime_implicants(
        prime_implicants, implicants_n=implicants_n
    )

    input_names = [
        name
        for column_index, name in enumerate(column_names)
        if column_index != output_column_index
    ]
    output_name = column_names[output_column_index]

    if as_latex:
        return render_latex(
            essential_prime_implicants,
            inputs_names=input_names,
            output_name=output_name,
        )

    return render(
        essential_prime_implicants, inputs_names=input_names, output_name=output_name
    )


def get_table_with_min_and_dont_care_terms(
    table: TTable, output_column_index: int
) -> TTable:
    return [row for row in table if row[output_column_index] in {"1", "X"}]


@dataclass(frozen=True)
class GroupRange:
    start: int
    end: int


@dataclass(frozen=True)
class Implicant:
    inputs: list[str]
    output: str
    minterms: list[int]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Implicant):
            raise RuntimeError("Implicants may only be compared to other implicants")

        if len(self.inputs) != len(other.inputs):
            raise RuntimeError(
                "Implicants must have the same number of inputs to be compared!"
            )

        return all(self[i] == other[i] for i in range(len(self.inputs)))

    def __add__(self, other: object) -> "Implicant":
        if not isinstance(other, Implicant):
            raise RuntimeError("Implicants may only be compared to other implicants")

        if len(self.inputs) != len(other.inputs):
            raise RuntimeError(
                "Implicants must have the same number of inputs to be compared!"
            )

        return self.combine(other)

    def __getitem__(self, index: int) -> str:
        return self.inputs[index]

    def __len__(self) -> int:
        return len(self.inputs)

    def combine(self, other: "Implicant") -> "Implicant":
        combine_index = self.get_combine_index(other)

        return Implicant(
            inputs=self.inputs[:combine_index]
            + ["-"]
            + self.inputs[combine_index + 1 :],
            output="X" if self.output == "X" and other.output == "X" else "1",
            minterms=self.minterms + other.minterms,
        )

    def get_combine_index(self, other: "Implicant") -> int:
        combine_index = None

        for i in range(len(self)):
            if self[i] != other[i]:
                if combine_index is None:
                    combine_index = i
                else:
                    raise ValueError

        if combine_index is None:
            raise RuntimeError("No implicants should be duplicated!")

        return combine_index


def convert_minterm_table_to_implicants(
    table: TTable, output_column_index: int
) -> list[Implicant]:
    implicants: list[Implicant] = []

    for minterm, row in enumerate(table):
        implicants.append(
            Implicant(
                inputs=[
                    cell
                    for column_index, cell in enumerate(row)
                    if column_index != output_column_index
                ],
                output=row[output_column_index],
                minterms=[minterm],
            )
        )

    return implicants


def get_sorted_implicants_and_group_ranges(
    implicants: list[Implicant]
) -> tuple[list[Implicant], list[GroupRange]]:
    number_of_1s_implicants_map: dict[int, list[Implicant]] = {}

    for implicant in implicants:
        number_of_1s = sum(1 if cell == "1" else 0 for cell in implicant.inputs)

        if number_of_1s in number_of_1s_implicants_map:
            number_of_1s_implicants_map[number_of_1s].append(implicant)
        else:
            number_of_1s_implicants_map[number_of_1s] = [implicant]

    sorted_implicants: list[Implicant] = []
    group_ranges: list[GroupRange] = []

    n = max(number_of_1s_implicants_map.keys())
    group_range_start = 0
    for i in range(1, n + 1):
        if i in number_of_1s_implicants_map:
            implicants = number_of_1s_implicants_map[i]

            sorted_implicants.extend(implicants)

            group_range_end = group_range_start + len(implicants)
            group_ranges.append(
                GroupRange(start=group_range_start, end=group_range_end)
            )

            group_range_start = group_range_end
        else:
            group_ranges.append(
                GroupRange(start=group_range_start, end=group_range_start)
            )

    return sorted_implicants, group_ranges


def get_combined_implicants_and_prime_implicants(
    implicants: list[Implicant], group_ranges: list[GroupRange]
) -> tuple[list[Implicant], list[Implicant]]:
    combined_implicants: list[Implicant] = []

    implicants_have_been_combined: list[bool] = [False] * len(implicants)

    for group_range1_index in range(len(group_ranges) - 1):
        group_range2_index = group_range1_index + 1

        group_range1 = group_ranges[group_range1_index]
        group_range2 = group_ranges[group_range2_index]

        for implicant1_index in range(group_range1.start, group_range1.end):
            implicant1 = implicants[implicant1_index]

            for implicant2_index in range(group_range2.start, group_range2.end):
                implicant2 = implicants[implicant2_index]

                try:
                    combined_implicant = implicant1 + implicant2

                    # Check if combination is duplicate.
                    if not any(
                        combined_implicant == implicant
                        for implicant in combined_implicants
                    ):
                        combined_implicants.append(combined_implicant)

                    implicants_have_been_combined[implicant1_index] = True
                    implicants_have_been_combined[implicant2_index] = True
                except ValueError:
                    pass

    prime_implacants = [
        implicant
        for implicant, has_been_combined in zip(
            implicants, implicants_have_been_combined
        )
        if (not has_been_combined) and implicant.output != "X"
    ]

    return combined_implicants, prime_implacants


def get_essential_prime_implicants(
    implicants: list[Implicant], implicants_n: int
) -> list[Implicant]:
    essential_prime_implicants: list[Implicant] = []

    chart = get_prime_implicants_chart(implicants, implicants_n)

    for column_index in range(implicants_n):
        implicant_index = None
        for row_index in range(len(implicants)):
            if chart[row_index][column_index] == 1:
                if implicant_index is None:
                    implicant_index = row_index
                else:
                    break  # type: ignore
        else:
            essential_prime_implicants.append(implicants[cast(int, implicant_index)])

    return essential_prime_implicants


def get_prime_implicants_chart(
    implicants: list[Implicant], implicants_n: int
) -> list[list[int]]:
    chart = [[0] * implicants_n for _ in range(len(implicants))]

    for row_index, implicant in enumerate(implicants):
        for minterm in implicant.minterms:
            chart[row_index][minterm] = 1

    return chart


def render(
    implicants: list[Implicant], inputs_names: list[str], output_name: str
) -> str:
    def resolve_negation(s: str) -> str:
        return f"~{s}"

    return _render(
        implicants=implicants,
        inputs_names=inputs_names,
        output_name=output_name,
        resolve_negation=resolve_negation,
    )


def render_latex(
    implicants: list[Implicant], inputs_names: list[str], output_name: str
) -> str:
    def resolve_negation(s: str) -> str:
        return re.sub(r"(.+)(^|_)?(.*)?", r"\\overline{\1}\2\3", s)

    return _render(
        implicants=implicants,
        inputs_names=inputs_names,
        output_name=output_name,
        resolve_negation=resolve_negation,
    )


def _render(
    implicants: list[Implicant],
    inputs_names: list[str],
    output_name: str,
    resolve_negation: Callable[[str], str],
) -> str:
    and_terms = []

    for implicant in implicants:
        and_term_literals = []
        for i, (input_name, input_) in enumerate(zip(inputs_names, implicant.inputs)):
            if input_ == "1":
                and_term_literals.append(input_name)
            elif input_ == "0":
                and_term_literals.append(resolve_negation(input_name))

        and_terms.append("".join(and_term_literals))

    return f"{output_name} = {' + '.join(and_terms)}"
